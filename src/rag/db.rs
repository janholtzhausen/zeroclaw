use anyhow::Result;
use deadpool_diesel::postgres::{Manager, Pool};
use deadpool_diesel::Runtime;
use diesel_migrations::{embed_migrations, EmbeddedMigrations, MigrationHarness};

pub const MIGRATIONS: EmbeddedMigrations =
    embed_migrations!("migrations/rag");

pub fn create_pool(db_url: &str) -> Result<Pool> {
    let manager = Manager::new(db_url.to_string(), Runtime::Tokio1);
    let pool = Pool::builder(manager).build()?;
    Ok(pool)
}

pub async fn run_migrations(pool: &Pool) -> Result<()> {
    let conn = pool.get().await?;
    let migration_result = conn
        .interact(|conn| conn.run_pending_migrations(MIGRATIONS).map(|_| ()))
        .await
    migration_result.map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(())
}

pub async fn init_pool_and_migrate(db_url: &str) -> Result<Pool> {
    let pool = create_pool(db_url)?;
    run_migrations(&pool).await?;
    Ok(pool)
}
