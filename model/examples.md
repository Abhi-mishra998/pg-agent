# TRAINING EXAMPLES

## Example 1: Slow Query with Sequential Scan

### INPUT:
- Signal: sequential_scan_detected on table "orders" (99% seq scan ratio)
- Metric: seq_scan_count=1523, table_rows=10234567
- Evidence: Query "SELECT * FROM orders WHERE customer_id = $1" taking 15s
- Root Cause: Missing index on customer_id column

### OUTPUT:
Diagnosis:
PostgreSQL is performing sequential scans on the orders table for queries filtering by customer_id, causing 15+ second query times and connection pool exhaustion.

Evidence:
- Sequential scan detected on orders table (seq_scan_ratio: 0.99)
- Table contains 10,234,567 rows
- Query "SELECT * FROM orders WHERE customer_id = $1" executing in 15,234ms
- Connection pool exhausted (198/200 connections active)
- No index exists on customer_id column

Root Cause:
Missing index on customer_id column forces PostgreSQL to read all 10M+ rows from disk/memory. The query was added recently without corresponding index maintenance.

Immediate Actions:
1. Create composite index on customer_id and created_at columns
2. Run EXPLAIN ANALYZE to verify index usage
3. Verify query performance improvement

Preventive Actions:
1. Add index creation review to code deployment process
2. Monitor sequential scan percentage in pg_stat_user_tables
3. Set up alerts for sequential scans on tables >100K rows
4. Review query patterns before deploying new features

CLI Commands:
```sql
-- Create index concurrently
CREATE INDEX CONCURRENTLY idx_orders_customer_created ON orders(customer_id, created_at);

-- Verify index is used
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM orders WHERE customer_id = 123;

-- Check sequential scan ratio
SELECT schemaname, tablename, seq_scan, idx_scan, 
       seq_scan::float / (seq_scan + idx_scan) as seq_scan_ratio
FROM pg_stat_user_tables WHERE relname = 'orders';
```

---

## Example 2: Blocking Transaction

### INPUT:
- Signal: blocking_detected
- Metric: wait_duration=120s, blocked_pid=34521, blocking_pid=33218
- Evidence: UPDATE blocking SELECT queries

### OUTPUT:
Diagnosis:
Active blocking transaction (PID 33218) holding RowExclusiveLock for 120+ seconds, blocking 5 other sessions and causing query timeouts.

Evidence:
- Blocking transaction active for 120 seconds
- 5 sessions blocked waiting for lock
- Blocking query: "BEGIN; UPDATE orders SET status = 'processing' WHERE customer_id = 12345;"
- Lock type: RowExclusiveLock
- Connection pool saturated (180/200 active)

Root Cause:
Long-running transaction holding lock while waiting for external resources (possibly application response or network). No statement_timeout configured to limit query duration.

Immediate Actions:
1. Identify blocking transaction details
2. Evaluate whether to terminate (pg_terminate_backend) or let complete
3. Cancel blocking query first (pg_cancel_backend) as less aggressive option

Preventive Actions:
1. Set statement_timeout to prevent long-running queries
2. Set idle_in_transaction_session_timeout
3. Implement connection pool query timeouts
4. Review application transaction handling
5. Ensure indexes exist for frequently updated columns

CLI Commands:
```sql
-- Identify blocking transaction
SELECT pid, usename, application_name, query, state, 
       now() - xact_start AS transaction_duration
FROM pg_stat_activity WHERE pid = 33218;

-- Check what blocker is waiting for
SELECT pid, wait_event_type, wait_event, state 
FROM pg_stat_activity WHERE pid = 33218;

-- Cancel blocking query (less disruptive)
SELECT pg_cancel_backend(33218);

-- Terminate only if necessary (rolls back transaction)
SELECT pg_terminate_backend(33218);

-- Monitor lock waits
SELECT blocked.pid AS blocked_pid, blocker.pid AS blocker_pid,
       blocked.query AS blocked_query, blocker.query AS blocker_query
FROM pg_stat_activity AS blocked
JOIN pg_stat_activity AS blocker ON blocked.pg_blocking_pids @> ARRAY[blocker.pid]
WHERE blocked.state = 'active';
```

---

## Example 3: Stale Statistics

### INPUT:
- Signal: stale_statistics detected
- Metric: days_since_analyze=14, rows_changed=50000
- Evidence: Query plan showing incorrect row estimates

### OUTPUT:
Diagnosis:
Table statistics are 14 days old, causing the PostgreSQL planner to make suboptimal execution plan decisions with severe row estimation errors.

Evidence:
- Statistics last updated 14 days ago (threshold: 7 days)
- 50,000 rows inserted/updated/deleted since last ANALYZE
- Query plan shows estimated 1000 rows vs actual 500,000 rows (500x error)
- Performance regression without schema changes

Root Cause:
ANALYZE not executed after significant data changes. Autovacuum thresholds not triggered due to low relative change percentage on large table. Stale statistics lead to incorrect cardinality estimates.

Immediate Actions:
1. Run ANALYZE on affected table
2. Verify statistics update completed
3. Check query plan improvement with EXPLAIN ANALYZE

Preventive Actions:
1. Set up alerting for tables with stale statistics (>7 days)
2. Configure appropriate autovacuum_analyze_scale_factor for high-churn tables
3. Run ANALYZE after bulk data operations
4. Monitor rows modified per table to detect statistics needs

CLI Commands:
```sql
-- Run ANALYZE to update statistics
ANALYZE VERBOSE table_name;

-- Check statistics age
SELECT relname, last_analyze, last_autoanalyze, n_live_tup, n_dead_tup
FROM pg_stat_user_tables WHERE relname = 'table_name';

-- Compare query plans before and after ANALYZE
EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM table_name WHERE filter_column = value;

-- Adjust autovacuum settings for high-churn table
ALTER TABLE table_name SET (autovacuum_analyze_scale_factor = 0.02, 
                            autovacuum_analyze_threshold = 1000);
```

---

## Example 4: High Dead Tuples

### INPUT:
- Signal: high_dead_tuples detected
- Metric: dead_tuple_ratio=0.45, n_dead_tup=1840000
- Evidence: VACUUM not keeping pace with DELETEs/UPDATEs

### OUTPUT:
Diagnosis:
High dead tuple ratio (45%) on orders table indicating VACUUM is not keeping pace with data modifications, causing wasted buffer cache and potential query performance degradation.

Evidence:
- Dead tuple ratio: 45% (threshold: 20%)
- 1,840,000 dead tuples on orders table
- Last vacuum: 10 days ago
- Sequential scan performance degrading due to bloat

Root Cause:
Heavy UPDATE/DELETE activity without adequate VACUUM. Autovacuum thresholds not triggered due to autovacuum_vacuum_scale_factor settings on large table. Dead tuples accumulate and bloat both table and indexes.

Immediate Actions:
1. Run VACUUM (VERBOSE) on affected table
2. Verify dead tuple ratio decreases
3. Check for query performance improvement

Preventive Actions:
1. Increase autovacuum frequency for high-churn tables
2. Lower autovacuum_vacuum_scale_factor for affected tables
3. Monitor dead tuple ratio with alerts
4. Schedule manual VACUUM during low-traffic periods if needed

CLI Commands:
```sql
-- Run VACUUM to reclaim space
VACUUM (VERBOSE, ANALYZE) table_name;

-- Check dead tuple ratio
SELECT relname, n_live_tup, n_dead_tup, 
       n_dead_tup::float / (n_dead_tup + n_live_tup) as dead_tuple_ratio
FROM pg_stat_user_tables WHERE relname = 'table_name';

-- Check autovacuum settings
SHOW autovacuum_vacuum_cost_delay;
SHOW autovacuum_naptime;

-- Adjust autovacuum for high-churn table
ALTER TABLE table_name SET (
    autovacuum_vacuum_scale_factor = 0.02,
    autovacuum_vacuum_threshold = 1000,
    autovacuum_analyze_scale_factor = 0.02,
    autovacuum_analyze_threshold = 1000
);
```

---

## Example 5: Insufficient work_mem

### INPUT:
- Signal: temp_file_spill_detected
- Metric: temp_blks_written=64000, work_mem=4MB
- Evidence: Query creating 1480 temp files, IO wait spikes

### OUTPUT:
Diagnosis:
Insufficient work_mem (4MB) causing query operations to spill to disk, generating 64,000+ temp blocks and causing significant I/O wait (28% of CPU time).

Evidence:
- work_mem: 4MB (recommended: 64MB+)
- temp_blks_written: 64,000 blocks
- 1,480 temp files created
- IO wait: 28% of CPU time
- Query latency increased 20x (500ms to 10s)

Root Cause:
work_mem too small for query memory requirements (sorting, hashing, hash joins). Operations that could fit in memory are spilling to disk, causing dramatic performance degradation.

Immediate Actions:
1. Increase work_mem for affected session
2. Verify temp file usage stops
3. Check query performance improvement

Preventive Actions:
1. Increase work_mem globally (consider per-operation limits)
2. Optimize queries to reduce memory requirements
3. Break large operations into smaller batches
4. Monitor temp file usage with pg_stat_statements

CLI Commands:
```sql
-- Increase work_mem for current session
SET work_mem = '128MB';

-- Check current work_mem setting
SHOW work_mem;

-- Monitor temp file usage
SELECT query, temp_blks_written, temp_blks_read,
       temp_blks_written::float / NULLIF(temp_blks_read, 0) as spill_ratio
FROM pg_stat_statements 
WHERE temp_blks_written > 0 
ORDER BY temp_blks_written DESC LIMIT 10;

-- Increase work_mem globally (requires reload)
ALTER SYSTEM SET work_mem = '64MB';
SELECT pg_reload_conf();

-- Check memory settings
SHOW shared_buffers;
SHOW effective_cache_size;
SHOW maintenance_work_mem;
```

---

## Example 6: Connection Pool Saturation

### INPUT:
- Signal: high_connection_usage
- Metric: active_connections=198/200, wait_events=47
- Evidence: Application timeouts, connection errors

### OUTPUT:
Diagnosis:
Database connection pool saturated (198/200 active connections) with 47 wait events, causing application timeouts and request failures.

Evidence:
- Active connections: 198/200 (99% utilization)
- 47 connection wait events
- Application reporting connection timeouts
- Error rate increased to 6.2%
- P95 latency spike to 4.8s

Root Cause:
Slow queries holding connections longer than normal, combined with increased request volume. No connection pool timeout settings. Connection leak possible in application.

Immediate Actions:
1. Identify long-running queries holding connections
2. Kill or wait for slow queries to complete
3. Restart application if connection leak suspected

Preventive Actions:
1. Set statement_timeout to limit query duration
2. Implement connection pool timeouts in application
3. Configure pgBouncer or similar connection pooler
4. Monitor and alert on connection utilization
5. Review application connection handling code

CLI Commands:
```sql
-- Check active connections
SELECT state, count(*) as connections
FROM pg_stat_activity 
WHERE datname = current_database()
GROUP BY state;

-- Find longest running queries
SELECT pid, usename, application_name, state, 
       now() - query_start as duration, query
FROM pg_stat_activity 
WHERE state != 'idle' 
ORDER BY duration DESC LIMIT 10;

-- Check wait events
SELECT wait_event_type, wait_event, count(*)
FROM pg_stat_activity 
WHERE state = 'active'
GROUP BY wait_event_type, wait_event;

-- Terminate longest running query (use with caution)
SELECT pg_terminate_backend(pid);
FROM pg_stat_activity 
WHERE state != 'idle' 
ORDER BY query_start ASC LIMIT 1;

-- Set statement timeout
SET statement_timeout = '30s';

-- Check max_connections
SHOW max_connections;
```

---

## Example 7: Index Bloat

### INPUT:
- Signal: high_index_bloat
- Metric: bloat_percent=128, index_size_mb=256
- Evidence: Index 2x larger than expected, scan performance degraded

### OUTPUT:
Diagnosis:
Index has 128% bloat (256MB actual vs 128MB expected), causing increased disk usage and degraded index scan performance.

Evidence:
- Index bloat estimate: 128%
- Current index size: 256MB
- Expected size: 128MB
- Index scan performance degrading
- Table has high update/delete activity

Root Cause:
Heavy UPDATE/DELETE activity causing index tuple fragmentation. VACUUM not reclaiming index space efficiently. Index needs reindexing to restore optimal performance.

Immediate Actions:
1. REINDEX CONCURRENTLY the affected index
2. Verify index size reduction
3. Check index scan performance

Preventive Actions:
1. Schedule regular REINDEX operations for high-churn indexes
2. Ensure autovacuum runs frequently enough
3. Consider index columns with low update frequency
4. Monitor index size growth over time

CLI Commands:
```sql
-- Check index bloat (requires pgstattuple extension)
SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) as size,
       pgstatindex(indexname::regclass) as stats
FROM pg_stat_user_indexes 
WHERE relname = 'table_name';

-- Check index size
SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) as size
FROM pg_stat_user_indexes 
WHERE relname = 'table_name';

-- REINDEX CONCURRENTLY (online, preferred)
REINDEX INDEX CONCURRENTLY index_name;

-- Standard REINDEX (locks writes, use during maintenance)
REINDEX INDEX index_name;

-- Verify index after reindex
SELECT indexname, pg_size_pretty(pg_relation_size(indexname::regclass)) as size
FROM pg_stat_user_indexes 
WHERE indexname = 'index_name';
```

