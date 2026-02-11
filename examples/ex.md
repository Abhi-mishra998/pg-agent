Example 1 — Index + Statistics + Deployment Regression

Input:

pg_stat_statements:
queryid: 1001
mean_exec_time: 18432 ms
calls: 210

EXPLAIN ANALYZE:
Seq Scan on customers (actual time=0.04..18211.32 rows=1)

pg_stat_user_tables:
relname: customers
last_analyze: 2025-11-15
n_dead_tup: 4212341

deployment_log:
2026-01-24 01:02 DROP INDEX customers_email_idx;


Expected Output:

Diagnosis:
Query latency increased due to index removal combined with stale statistics.

Evidence:

Sequential scan observed on customers table

Index drop recorded in deployment log

High dead tuple count with outdated analyze timestamp

Root Cause:
Critical index removal without statistics refresh.

Immediate Actions:

Recreate missing index

Run ANALYZE on affected table

Preventive Actions:

Enforce schema-change reviews

Run ANALYZE after index changes

CLI Commands:

CREATE INDEX CONCURRENTLY customers_email_idx ON customers(email);
ANALYZE customers;

Example 2 — Lock Contention + Idle Transaction + Application Bug

Input:

pg_stat_activity:
pid: 2241
state: idle in transaction
xact_start: 01:12:01

pg_stat_activity:
pid: 2299
wait_event_type: Lock
wait_event: transactionid

application_log:
OrderService timeout after 30s


Expected Output:

Diagnosis:
Query blocking caused by idle-in-transaction session from application.

Evidence:

Blocking PID idle in transaction

Waiting PID blocked on transactionid lock

Application timeouts reported

Root Cause:
Application failed to close transactions.

Immediate Actions:

Terminate blocking session

Notify application team

Preventive Actions:

Set idle_in_transaction_session_timeout

Add transaction lifecycle checks

CLI Commands:

SELECT pid, state, xact_start FROM pg_stat_activity;

Example 3 — Temp Spill + work_mem + Disk IO Saturation

Input:

pg_stat_statements:
queryid: 3003
temp_blks_written: 524288
mean_exec_time: 22451 ms

postgresql.conf:
work_mem = 4MB

OS metrics:
disk utilization: 94%


Expected Output:

Diagnosis:
Query slowdown caused by temp file spills leading to disk IO saturation.

Evidence:

High temp blocks written

Low work_mem configuration

Disk utilization near saturation

Root Cause:
Insufficient work_mem for query workload.

Immediate Actions:

Increase work_mem for session

Monitor temp file creation

Preventive Actions:

Review memory tuning guidelines

Alert on temp spill thresholds

CLI Commands:

SET work_mem = '128MB';

Example 4 — Autovacuum Starvation + Write Surge + IO Pressure

Input:

pg_stat_user_tables:
relname: events
n_dead_tup: 15234123
last_autovacuum: NULL

pg_stat_activity:
state: active
count: 420

OS metrics:
disk io latency: 42ms


Expected Output:

Diagnosis:
Query performance degraded due to autovacuum starvation under heavy write load.

Evidence:

Extremely high dead tuple count

No recent autovacuum activity

Elevated disk IO latency

Root Cause:
Autovacuum unable to keep up with write volume.

Immediate Actions:

Run manual VACUUM

Monitor IO load during cleanup

Preventive Actions:

Tune autovacuum thresholds

Separate write-heavy workloads

CLI Commands:

VACUUM (VERBOSE) events;

Example 5 — Connection Storm + CPU Saturation + Missing Pooler

Input:

pg_stat_activity:
count: 950

postgresql.conf:
max_connections = 1000

OS metrics:
CPU usage: 99%


Expected Output:

Diagnosis:
CPU saturation caused by excessive concurrent connections.

Evidence:

Nearly all connections in use

CPU utilization at 99%

Root Cause:
Lack of connection pooling.

Immediate Actions:

Throttle application connections

Monitor connection states

Preventive Actions:

Deploy PgBouncer

Reduce max_connections

CLI Commands:

SELECT count(*) FROM pg_stat_activity;

Example 6 — Backup Job + Checkpoint Storm + Query Latency

Input:

postgresql.log:
checkpoint complete: wrote 112341 buffers

OS process:
pg_basebackup running

pg_stat_statements:
mean_exec_time: 16321 ms


Expected Output:

Diagnosis:
Query latency caused by IO contention from backup and frequent checkpoints.

Evidence:

Active base backup process

Large checkpoint writes

Increased query execution time

Root Cause:
Backup job overlapping with peak workload.

Immediate Actions:

Pause or reschedule backup

Monitor IO metrics

Preventive Actions:

Move backups off-peak

Tune checkpoint settings

CLI Commands:

SELECT * FROM pg_stat_bgwriter;

Example 7 — Plan Regression After PostgreSQL Upgrade

Input:

pg_stat_statements:
queryid: 7711
mean_exec_time_before: 412 ms
mean_exec_time_after: 8123 ms

version_log:
Upgrade from PostgreSQL 13 to 15


Expected Output:

Diagnosis:
Query plan regression following PostgreSQL version upgrade.

Evidence:

Significant execution time increase post-upgrade

Recorded version change

Root Cause:
Planner behavior change after upgrade.

Immediate Actions:

Review EXPLAIN ANALYZE output

Test alternate query plans

Preventive Actions:

Run plan regression tests pre-upgrade

Capture baseline plans

CLI Commands:

EXPLAIN ANALYZE <query>;

Example 8 — Schema Change + Bloat + IO Waits

Input:

deployment_log:
ALTER TABLE payments ADD COLUMN notes text;

pg_stat_user_tables:
n_dead_tup: 8345123

pg_stat_activity:
wait_event_type: IO


Expected Output:

Diagnosis:
Performance degradation due to bloat after schema change.

Evidence:

Schema alteration logged

High dead tuple count

IO wait events observed

Root Cause:
Bloat accumulation post-DDL.

Immediate Actions:

Run VACUUM and ANALYZE

Monitor IO waits

Preventive Actions:

Post-DDL maintenance procedures

Bloat monitoring alerts

CLI Commands:

VACUUM (ANALYZE) payments;

Example 9 — Replica Lag + WAL Surge + Reporting Load

Input:

pg_stat_replication:
replay_lag: 12min

pg_stat_statements:
bulk_report_query temp_blks_written: 412341


Expected Output:

Diagnosis:
Replica lag caused by WAL surge from reporting workload.

Evidence:

High replication lag

Temp-heavy reporting queries

Root Cause:
Reporting workload generating excessive WAL.

Immediate Actions:

Throttle reporting jobs

Monitor replication slots

Preventive Actions:

Offload reporting to replica

Limit bulk operations

CLI Commands:

SELECT * FROM pg_stat_replication;

Example 10 — Memory Pressure + Kernel Swapping + Query Timeouts

Input:

OS metrics:
swap usage: 18GB
memory free: 2%

pg_stat_activity:
wait_event_type: IO
wait_event: DataFileRead


Expected Output:

Diagnosis:
Query timeouts caused by memory pressure leading to disk swapping.

Evidence:

High swap usage

Low free memory

IO wait events

Root Cause:
System memory exhaustion causing swap thrashing.

Immediate Actions:

Reduce concurrent workload

Monitor swap usage

Preventive Actions:

Increase system memory

Tune PostgreSQL memory parameters

CLI Commands:

free -m
SELECT pid, wait_event FROM pg_stat_activity;