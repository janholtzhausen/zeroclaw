use crate::rag::db;
use anyhow::{Context, Result};
use deadpool_diesel::postgres::Pool;
use diesel::RunQueryDsl;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone)]
pub struct EmbeddingClient {
    client: Client,
    api_key: String,
    base_url: String,
    model: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct EmbeddingRequest<'a> {
    input: &'a str,
    model: &'a str,
    input_type: &'a str,
}

impl EmbeddingClient {
    pub fn new(api_key: &str, base_url: &str, model: &str) -> Result<Self> {
        let client = Client::builder().build()?;
        Ok(Self {
            client,
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        })
    }

    pub async fn embed(&self, input: &str, input_type: &str) -> Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.base_url);
        let req = EmbeddingRequest {
            input,
            model: &self.model,
            input_type,
        };

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&req)
            .send()
            .await
            .context("failed to call embedding API")?
            .error_for_status()
            .context("embedding API returned error status")?;

        let payload: EmbeddingResponse = resp
            .json()
            .await
            .context("invalid embedding API response")?;

        let first = payload
            .data
            .into_iter()
            .next()
            .context("embedding API returned empty data")?;

        if first.embedding.len() != 1024 {
            anyhow::bail!(
                "embedding dimension mismatch: expected 1024, got {}",
                first.embedding.len()
            );
        }

        Ok(first.embedding)
    }
}

#[derive(Clone)]
pub struct PgVectorRagStore {
    pool: Pool,
    embedding: EmbeddingClient,
    similarity_threshold: f32,
}

impl PgVectorRagStore {
    pub fn new_without_migrations(
        db_url: &str,
        embedding_api_key: &str,
        embedding_base_url: &str,
        embedding_model: &str,
        similarity_threshold: f32,
    ) -> Result<Self> {
        let pool = db::create_pool(db_url)?;
        let embedding =
            EmbeddingClient::new(embedding_api_key, embedding_base_url, embedding_model)?;
        Ok(Self {
            pool,
            embedding,
            similarity_threshold,
        })
    }

    pub async fn new(
        db_url: &str,
        embedding_api_key: &str,
        embedding_base_url: &str,
        embedding_model: &str,
        similarity_threshold: f32,
    ) -> Result<Self> {
        let pool = db::init_pool_and_migrate(db_url).await?;
        let embedding =
            EmbeddingClient::new(embedding_api_key, embedding_base_url, embedding_model)?;
        Ok(Self {
            pool,
            embedding,
            similarity_threshold,
        })
    }

    pub async fn ingest_web_content(&self, source_url: &str, body: &str) -> Result<()> {
        let trimmed = body.trim();
        if trimmed.is_empty() {
            return Ok(());
        }

        let embedding = self.embedding.embed(trimmed, "passage").await?;
        let embedding_sql = vector_to_sql(&embedding);
        let source_url = source_url.to_string();
        let content = trimmed.to_string();

        let conn = self.pool.get().await?;
        conn.interact(move |conn| {
            let source_id = Uuid::new_v4();
            let source_id_str = source_id.to_string();
            let chunk_id = Uuid::new_v4().to_string();
            let escaped_url = sql_escape(&source_url);
            let escaped_content = sql_escape(&content);
            let stmt = format!(
                "INSERT INTO rag_sources (id, source_type, source_url, title, metadata) \
                 VALUES ('{source_id_str}', 'web', '{escaped_url}', NULL, '{{}}'::jsonb);\
                 INSERT INTO rag_chunks \
                 (id, source_id, chunk_index, content, heading_context, embedding, token_count, metadata) \
                 VALUES ('{chunk_id}', '{source_id_str}', 0, '{escaped_content}', NULL, '{embedding_sql}'::vector, NULL, '{{}}'::jsonb);"
            );
            diesel::sql_query(stmt).execute(conn).map(|_| ())
        })
        .await??;

        Ok(())
    }

    pub async fn retrieve_context(&self, query: &str, top_k: usize) -> Result<String> {
        if query.trim().is_empty() || top_k == 0 {
            return Ok(String::new());
        }

        let embedding = self.embedding.embed(query, "query").await?;
        let embedding_sql = vector_to_sql(&embedding);
        let threshold = self.similarity_threshold;

        let conn = self.pool.get().await?;
        let rows = conn
            .interact(move |conn| -> Result<Vec<(String, String, String)>> {
                use diesel::sql_types::{Double, Nullable, Text};

                #[derive(diesel::QueryableByName)]
                struct RagRow {
                    #[diesel(sql_type = Text)]
                    content: String,
                    #[diesel(sql_type = Nullable<Text>)]
                    heading_context: Option<String>,
                    #[diesel(sql_type = Nullable<Text>)]
                    source_url: Option<String>,
                    #[diesel(sql_type = Double)]
                    similarity: f64,
                }

                let stmt = format!(
                    "SELECT c.content, c.heading_context, s.source_url, 1 - (c.embedding <=> '{embedding_sql}'::vector) AS similarity \
                     FROM rag_chunks c \
                     INNER JOIN rag_sources s ON s.id = c.source_id \
                     WHERE s.active = TRUE \
                     ORDER BY c.embedding <=> '{embedding_sql}'::vector \
                     LIMIT {}",
                    top_k
                );

                let results = diesel::sql_query(stmt)
                    .load::<RagRow>(conn)
                    .context("failed to query rag_chunks")?;

                let filtered = results
                    .into_iter()
                    .filter(|row| row.similarity as f32 >= threshold)
                    .map(|row| {
                        (
                            row.heading_context.unwrap_or_else(|| "context".to_string()),
                            row.source_url.unwrap_or_else(|| "unknown".to_string()),
                            row.content,
                        )
                    })
                    .collect::<Vec<_>>();

                Ok(filtered)
            })
            .await??;

        if rows.is_empty() {
            return Ok(String::new());
        }

        let mut context = String::from(
            "<context>\nThe following information was retrieved from official Immigration New Zealand sources.\nUse this to inform your response. Always cite the source section when referencing specific policies.\n\n",
        );

        for (heading, url, content) in rows {
            context.push_str(&format!(
                "[Source: {}]\n[URL: {}]\n<content>\n{}\n</content>\n\n",
                heading, url, content
            ));
        }
        context.push_str("</context>");

        Ok(context)
    }
}

fn vector_to_sql(values: &[f32]) -> String {
    let inner = values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{inner}]")
}

fn sql_escape(input: &str) -> String {
    input.replace('"', "\"\"").replace('\'', "''")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_sql_format() {
        let got = vector_to_sql(&[1.0, 2.5, -3.0]);
        assert_eq!(got, "[1,2.5,-3]");
    }

    #[test]
    fn context_empty_for_empty_query() {
        let store = PgVectorRagStore {
            pool: db::create_pool("postgres://localhost/invalid")
                .expect("pool creation should not connect"),
            embedding: EmbeddingClient::new(
                "k",
                "https://example.com/v1",
                "nvidia/nv-embedqa-e5-v5",
            )
            .expect("embedding client"),
            similarity_threshold: 0.3,
        };

        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let result = rt
            .block_on(store.retrieve_context("", 5))
            .expect("empty query succeeds");
        assert!(result.is_empty());
    }
}
