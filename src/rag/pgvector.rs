use anyhow::{Context, Result};
use postgres::{Client, NoTls};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

use parking_lot::Mutex;

const MAX_EMBEDDING_BATCH: usize = 50;
const EMBEDDING_DIMS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingInputType {
    Query,
    Passage,
}

impl EmbeddingInputType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Passage => "passage",
        }
    }
}

#[derive(Debug, Clone)]
pub struct EmbeddingClient {
    http: reqwest::Client,
    api_key: String,
    model: String,
    endpoint: String,
    max_retries: u8,
}

impl EmbeddingClient {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            http: crate::config::build_runtime_proxy_client("rag.embeddings"),
            api_key: api_key.to_string(),
            model: model.to_string(),
            endpoint: "https://integrate.api.nvidia.com/v1/embeddings".to_string(),
            max_retries: 3,
        }
    }

    pub async fn embed(
        &self,
        texts: &[String],
        input_type: EmbeddingInputType,
    ) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(MAX_EMBEDDING_BATCH) {
            let vectors = self.embed_batch(batch, input_type).await?;
            out.extend(vectors);
        }

        Ok(out)
    }

    async fn embed_batch(
        &self,
        texts: &[String],
        input_type: EmbeddingInputType,
    ) -> Result<Vec<Vec<f32>>> {
        let body = serde_json::json!({
            "input": texts,
            "model": self.model,
            "input_type": input_type.as_str(),
            "encoding_format": "float",
            "truncate": "END"
        });

        for attempt in 0..=self.max_retries {
            let response = self
                .http
                .post(&self.endpoint)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .await
                .context("failed to call NVIDIA embedding API")?;

            if response.status().is_success() {
                let payload: serde_json::Value = response
                    .json()
                    .await
                    .context("failed to decode NVIDIA embedding response")?;

                let data = payload
                    .get("data")
                    .and_then(serde_json::Value::as_array)
                    .ok_or_else(|| anyhow::anyhow!("embedding response missing data array"))?;

                let mut vectors = Vec::with_capacity(data.len());
                for item in data {
                    let embedding = item
                        .get("embedding")
                        .and_then(serde_json::Value::as_array)
                        .ok_or_else(|| anyhow::anyhow!("embedding item missing embedding array"))?;
                    #[allow(clippy::cast_possible_truncation)]
                    let vector: Vec<f32> = embedding
                        .iter()
                        .filter_map(|value| value.as_f64().map(|v| v as f32))
                        .collect();
                    if vector.len() != EMBEDDING_DIMS {
                        anyhow::bail!(
                            "unexpected embedding dimensions: got {}, expected {EMBEDDING_DIMS}",
                            vector.len()
                        );
                    }
                    vectors.push(vector);
                }

                return Ok(vectors);
            }

            let status = response.status();
            let is_retryable = status.as_u16() == 429 || status.is_server_error();
            if !is_retryable || attempt == self.max_retries {
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!("NVIDIA embedding API error {status}: {body}");
            }

            let delay_ms = 250_u64.saturating_mul(2_u64.pow(u32::from(attempt)));
            sleep(Duration::from_millis(delay_ms)).await;
        }

        anyhow::bail!("unreachable embedding retry path")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    pub content: String,
    pub heading_context: Option<String>,
    pub chunk_index: usize,
    pub approximate_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IngestMetadata {
    pub source_url: Option<String>,
    pub title: Option<String>,
    pub source_type: String,
    pub metadata_json: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub content: String,
    pub heading_context: Option<String>,
    pub source_url: Option<String>,
    pub source_title: Option<String>,
    pub similarity_score: f64,
}

#[derive(Clone)]
pub struct PgVectorRagStore {
    client: Arc<Mutex<Client>>,
    schema: String,
    pub similarity_threshold: f64,
}

impl PgVectorRagStore {
    pub fn new(db_url: &str, schema: &str, similarity_threshold: f64) -> Result<Self> {
        let schema_ident = format!("\"{}\"", schema);
        let client = Self::initialize_client(db_url.to_string(), schema_ident)?;

        Ok(Self {
            client: Arc::new(Mutex::new(client)),
            schema: schema.to_string(),
            similarity_threshold,
        })
    }

    fn initialize_client(db_url: String, schema_ident: String) -> Result<Client> {
        let init_handle = std::thread::Builder::new()
            .name("pgvector-rag-init".to_string())
            .spawn(move || -> Result<Client> {
                let config: postgres::Config = db_url
                    .parse()
                    .context("invalid PostgreSQL connection URL")?;

                let mut client = config
                    .connect(NoTls)
                    .context("failed to connect PostgreSQL for RAG")?;

                Self::init_schema(&mut client, &schema_ident)?;
                Ok(client)
            })
            .context("failed to spawn PostgreSQL RAG initializer thread")?;

        let init_result = init_handle
            .join()
            .map_err(|_| anyhow::anyhow!("PostgreSQL RAG initializer thread panicked"))?;

        init_result
    }

    fn init_schema(client: &mut Client, schema_ident: &str) -> Result<()> {
        client
            .batch_execute(&format!(
                "
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE EXTENSION IF NOT EXISTS pgcrypto;
                CREATE SCHEMA IF NOT EXISTS {schema_ident};

                CREATE TABLE IF NOT EXISTS {schema_ident}.rag_sources (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_type TEXT NOT NULL,
                    source_url TEXT,
                    title TEXT,
                    crawled_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_updated TIMESTAMPTZ,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    active BOOLEAN NOT NULL DEFAULT TRUE
                );

                CREATE TABLE IF NOT EXISTS {schema_ident}.rag_chunks (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    source_id UUID NOT NULL REFERENCES {schema_ident}.rag_sources(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    heading_context TEXT,
                    embedding vector(1024) NOT NULL,
                    token_count INTEGER,
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(source_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_rag_chunks_embedding
                    ON {schema_ident}.rag_chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 200);

                CREATE INDEX IF NOT EXISTS idx_rag_chunks_source_id ON {schema_ident}.rag_chunks(source_id);
                CREATE INDEX IF NOT EXISTS idx_rag_sources_crawled_at ON {schema_ident}.rag_sources(crawled_at);
                "
            ))
            .context("failed to initialize pgvector RAG schema")?;

        Ok(())
    }

    pub async fn ingest(
        &self,
        embedder: &EmbeddingClient,
        content: &str,
        chunk_size_tokens: usize,
        chunk_overlap_tokens: usize,
        metadata: &IngestMetadata,
    ) -> Result<bool> {
        let chunks = chunk_text(content, chunk_size_tokens, chunk_overlap_tokens);
        if chunks.is_empty() {
            return Ok(false);
        }

        let payloads: Vec<String> = chunks.iter().map(|chunk| chunk.content.clone()).collect();
        let embeddings = embedder
            .embed(&payloads, EmbeddingInputType::Passage)
            .await?;

        if embeddings.len() != chunks.len() {
            anyhow::bail!("embedding count does not match chunk count");
        }

        let source_type = metadata.source_type.trim();
        if source_type.is_empty() {
            anyhow::bail!("source_type must not be empty");
        }

        let schema = self.schema.clone();
        let source_url = metadata.source_url.clone();
        let title = metadata.title.clone();
        let source_type_owned = source_type.to_string();
        let metadata_json = metadata
            .metadata_json
            .clone()
            .unwrap_or_else(|| "{}".to_string());
        let content_hash = hash_content(content);

        let client = self.client.clone();
        tokio::task::spawn_blocking(move || -> Result<bool> {
            let mut client = client.lock();
            let mut tx = client.transaction().context("failed to start RAG transaction")?;

            let src_query = format!(
                "
                SELECT id::text, COALESCE(metadata->>'content_hash', '')
                FROM \"{schema}\".rag_sources
                WHERE source_type = $1
                  AND source_url IS NOT DISTINCT FROM $2
                  AND title IS NOT DISTINCT FROM $3
                LIMIT 1
                "
            );

            let existing = tx
                .query_opt(&src_query, &[&source_type_owned, &source_url, &title])
                .context("failed to fetch existing rag source")?;

            let source_id: String;
            let changed: bool;

            if let Some(row) = existing {
                source_id = row.get(0);
                let existing_hash: String = row.get(1);
                changed = existing_hash != content_hash;
                if !changed {
                    tx.rollback().context("failed to rollback no-op transaction")?;
                    return Ok(false);
                }

                let new_metadata = merge_content_hash(&metadata_json, &content_hash);
                let update_sql = format!(
                    "
                    UPDATE \"{schema}\".rag_sources
                    SET last_updated = NOW(), metadata = $1::jsonb, active = TRUE
                    WHERE id = $2::uuid
                    "
                );
                tx.execute(&update_sql, &[&new_metadata, &source_id])
                    .context("failed to update rag source")?;

                let delete_sql = format!("DELETE FROM \"{schema}\".rag_chunks WHERE source_id = $1::uuid");
                tx.execute(&delete_sql, &[&source_id])
                    .context("failed to delete previous rag chunks")?;
            } else {
                changed = true;
                let new_metadata = merge_content_hash(&metadata_json, &content_hash);
                let insert_sql = format!(
                    "
                    INSERT INTO \"{schema}\".rag_sources (source_type, source_url, title, metadata, last_updated)
                    VALUES ($1, $2, $3, $4::jsonb, NOW())
                    RETURNING id::text
                    "
                );
                let row = tx
                    .query_one(&insert_sql, &[&source_type_owned, &source_url, &title, &new_metadata])
                    .context("failed to insert rag source")?;
                source_id = row.get(0);
            }

            let insert_chunk_sql = format!(
                "
                INSERT INTO \"{schema}\".rag_chunks
                    (source_id, chunk_index, content, heading_context, embedding, token_count)
                VALUES
                    ($1::uuid, $2, $3, $4, $5::vector, $6)
                "
            );

            for (idx, chunk) in chunks.iter().enumerate() {
                let embedding_literal = vector_literal(&embeddings[idx]);
                let chunk_idx = i32::try_from(chunk.chunk_index).context("chunk index overflow")?;
                let token_count = i32::try_from(chunk.approximate_tokens).context("token count overflow")?;
                tx.execute(
                    &insert_chunk_sql,
                    &[&source_id, &chunk_idx, &chunk.content, &chunk.heading_context, &embedding_literal, &token_count],
                )
                .context("failed to insert rag chunk")?;
            }

            tx.commit().context("failed to commit RAG ingestion transaction")?;
            Ok(changed)
        })
        .await?
    }

    pub async fn retrieve(
        &self,
        embedder: &EmbeddingClient,
        query: &str,
        top_k: usize,
        source_type: Option<&str>,
        only_active: bool,
    ) -> Result<Vec<RetrievalResult>> {
        let query_vec = embedder
            .embed(&[query.to_string()], EmbeddingInputType::Query)
            .await?
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("query embedding is empty"))?;
        let vector = vector_literal(&query_vec);

        let schema = self.schema.clone();
        let similarity_threshold = self.similarity_threshold;
        let source_type_filter = source_type.map(str::to_string);
        let active_filter = only_active;

        let client = self.client.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<RetrievalResult>> {
            let mut client = client.lock();
            let limit = i64::try_from(top_k).context("top_k is too large")?;

            let sql = format!(
                "
                SELECT c.content,
                       c.heading_context,
                       s.source_url,
                       s.title,
                       1 - (c.embedding <=> $1::vector) AS similarity
                FROM \"{schema}\".rag_chunks c
                JOIN \"{schema}\".rag_sources s ON c.source_id = s.id
                WHERE ($2::boolean = FALSE OR s.active = TRUE)
                  AND ($3::text IS NULL OR s.source_type = $3)
                ORDER BY c.embedding <=> $1::vector
                LIMIT $4
                "
            );

            let rows = client
                .query(
                    &sql,
                    &[&vector, &active_filter, &source_type_filter, &limit],
                )
                .context("failed RAG retrieval query")?;

            let mut results = Vec::with_capacity(rows.len());
            for row in rows {
                let score: f64 = row.get(4);
                if score < similarity_threshold {
                    continue;
                }

                results.push(RetrievalResult {
                    content: row.get(0),
                    heading_context: row.get(1),
                    source_url: row.get(2),
                    source_title: row.get(3),
                    similarity_score: score,
                });
            }

            Ok(results)
        })
        .await?
    }
}

pub fn format_retrieval_context(results: &[RetrievalResult]) -> String {
    if results.is_empty() {
        return String::new();
    }

    let mut out = String::from(
        "<context>\nThe following information was retrieved from official Immigration New Zealand sources.\nUse this to inform your response. Always cite the source section when referencing specific policies.\n\n",
    );

    for result in results {
        let heading = result
            .heading_context
            .as_deref()
            .unwrap_or("Unknown section");
        out.push_str(&format!("[Source: {heading}]\n"));
        if let Some(url) = &result.source_url {
            out.push_str(&format!("[URL: {url}]\n"));
        }
        if let Some(title) = &result.source_title {
            out.push_str(&format!("[Title: {title}]\n"));
        }
        out.push_str(&format!(
            "[Similarity: {:.3}]\n<content>\n{}\n</content>\n\n",
            result.similarity_score, result.content
        ));
    }

    out.push_str("</context>\n\n");
    out
}

pub fn chunk_text(
    content: &str,
    chunk_size_tokens: usize,
    chunk_overlap_tokens: usize,
) -> Vec<TextChunk> {
    if content.trim().is_empty() {
        return Vec::new();
    }

    let max_chars = chunk_size_tokens.saturating_mul(4);
    let overlap_chars = chunk_overlap_tokens.saturating_mul(4);

    let mut heading_stack: Vec<String> = Vec::new();
    let mut blocks: Vec<(Option<String>, String)> = Vec::new();
    let mut paragraph = String::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if let Some((level, heading_text)) = parse_heading(trimmed) {
            flush_paragraph(&mut paragraph, &heading_stack, &mut blocks);
            if heading_stack.len() >= level {
                heading_stack.truncate(level.saturating_sub(1));
            }
            heading_stack.push(heading_text.to_string());
            continue;
        }

        if trimmed.is_empty() {
            flush_paragraph(&mut paragraph, &heading_stack, &mut blocks);
            continue;
        }

        if !paragraph.is_empty() {
            paragraph.push(' ');
        }
        paragraph.push_str(trimmed);
    }

    flush_paragraph(&mut paragraph, &heading_stack, &mut blocks);

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_heading: Option<String> = None;

    for (heading, block) in blocks {
        let target_heading = heading.clone();
        let needs_heading_reset = current_heading != target_heading;

        if needs_heading_reset && !current.is_empty() {
            push_chunk(&mut chunks, &current, current_heading.clone());
            current.clear();
        }

        current_heading = target_heading;
        for sentence in split_sentences(&block) {
            if sentence.len() > max_chars {
                if !current.is_empty() {
                    push_chunk(&mut chunks, &current, current_heading.clone());
                    current.clear();
                }
                for fragment in split_large_sentence(&sentence, max_chars) {
                    push_chunk(&mut chunks, &fragment, current_heading.clone());
                }
                continue;
            }

            let projected_len = current.len() + sentence.len() + usize::from(!current.is_empty());
            if projected_len > max_chars && !current.is_empty() {
                push_chunk(&mut chunks, &current, current_heading.clone());
                current = tail_overlap(&current, overlap_chars);
            }

            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(&sentence);
        }
    }

    if !current.is_empty() {
        push_chunk(&mut chunks, &current, current_heading);
    }

    for (idx, chunk) in chunks.iter_mut().enumerate() {
        chunk.chunk_index = idx;
    }

    chunks
}

fn parse_heading(line: &str) -> Option<(usize, &str)> {
    let hashes = line.chars().take_while(|ch| *ch == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }

    let heading = line[hashes..].trim();
    if heading.is_empty() {
        return None;
    }

    Some((hashes, heading))
}

fn flush_paragraph(
    paragraph: &mut String,
    heading_stack: &[String],
    blocks: &mut Vec<(Option<String>, String)>,
) {
    if paragraph.is_empty() {
        return;
    }

    let heading = if heading_stack.is_empty() {
        None
    } else {
        Some(heading_stack.join(" > "))
    };

    blocks.push((heading, std::mem::take(paragraph)));
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut start = 0usize;
    let chars: Vec<char> = text.chars().collect();

    for (idx, ch) in chars.iter().enumerate() {
        if matches!(ch, '.' | '!' | '?') {
            let next_is_boundary = chars.get(idx + 1).is_none_or(|next| next.is_whitespace());
            if next_is_boundary {
                let sentence: String = chars[start..=idx].iter().collect();
                let trimmed = sentence.trim();
                if !trimmed.is_empty() {
                    sentences.push(trimmed.to_string());
                }
                start = idx + 1;
            }
        }
    }

    if start < chars.len() {
        let tail: String = chars[start..].iter().collect();
        let trimmed = tail.trim();
        if !trimmed.is_empty() {
            sentences.push(trimmed.to_string());
        }
    }

    sentences
}

fn split_large_sentence(sentence: &str, max_chars: usize) -> Vec<String> {
    if sentence.len() <= max_chars {
        return vec![sentence.to_string()];
    }

    let mut out = Vec::new();
    let mut start = 0;
    while start < sentence.len() {
        let end = cmp::min(start + max_chars, sentence.len());
        out.push(sentence[start..end].trim().to_string());
        start = end;
    }
    out
}

fn tail_overlap(text: &str, overlap_chars: usize) -> String {
    if overlap_chars == 0 || text.len() <= overlap_chars {
        return text.to_string();
    }

    let start = text.len() - overlap_chars;
    text[start..].trim_start().to_string()
}

fn push_chunk(chunks: &mut Vec<TextChunk>, content: &str, heading_context: Option<String>) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }

    chunks.push(TextChunk {
        content: trimmed.to_string(),
        heading_context,
        chunk_index: chunks.len(),
        approximate_tokens: approximate_tokens(trimmed),
    });
}

fn approximate_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    chars.div_ceil(4)
}

fn hash_content(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn merge_content_hash(metadata_json: &str, content_hash: &str) -> String {
    let mut value = serde_json::from_str::<serde_json::Value>(metadata_json)
        .unwrap_or_else(|_| serde_json::json!({}));

    let map = value.as_object_mut();
    if let Some(obj) = map {
        obj.insert(
            "content_hash".to_string(),
            serde_json::Value::String(content_hash.to_string()),
        );
    } else {
        value = serde_json::json!({ "content_hash": content_hash });
    }

    value.to_string()
}

fn vector_literal(vector: &[f32]) -> String {
    let payload = vector
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<String>>()
        .join(",");
    format!("[{payload}]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunker_tracks_heading_context_and_overlap() {
        let text = "# Immigration Instructions\n\n## Residence\n\nCharacter requirements apply. Applicants must provide police certificates.\n\n## Work\n\nWork visa rules differ by pathway.";
        let chunks = chunk_text(text, 10, 2);

        assert!(!chunks.is_empty());
        assert!(chunks.iter().any(|chunk| chunk.heading_context.as_deref()
            == Some("Immigration Instructions > Residence")));
        assert!(chunks.iter().all(|chunk| chunk.approximate_tokens > 0));
        assert_eq!(chunks[0].chunk_index, 0);
    }

    #[test]
    fn chunker_never_returns_empty_content() {
        let text = "\n\n# Heading\n\n";
        let chunks = chunk_text(text, 100, 10);
        assert!(chunks.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn ingest_and_retrieve_roundtrip_postgres() {
        let Ok(db_url) = std::env::var("ZEROCLAW_TEST_POSTGRES_URL") else {
            return;
        };
        let Ok(api_key) = std::env::var("ZEROCLAW_TEST_NVIDIA_EMBEDDING_API_KEY") else {
            return;
        };

        let store = PgVectorRagStore::new(&db_url, "public", 0.0).expect("store should initialize");
        let embedder = EmbeddingClient::new(&api_key, "nvidia/nv-embedqa-e5-v5");

        let ingested = store
            .ingest(
                &embedder,
                "# Residence\nCharacter requirements include police certificates.",
                50,
                5,
                &IngestMetadata {
                    source_url: Some("https://example.com/inz/residence".to_string()),
                    title: Some("Residence Character".to_string()),
                    source_type: "web".to_string(),
                    metadata_json: None,
                },
            )
            .await
            .expect("ingest should succeed");

        assert!(ingested);

        let results = store
            .retrieve(
                &embedder,
                "What are residence character requirements?",
                5,
                Some("web"),
                true,
            )
            .await
            .expect("retrieve should succeed");

        assert!(!results.is_empty());
    }
}
