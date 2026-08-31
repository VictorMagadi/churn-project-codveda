-- =============================================================================
-- Level 2 - Task 2: SQL for Business Analytics
-- Codveda Business Analytics Internship
--
-- Written for SQLite (used by run_sql_demo.py, no server needed).
-- Notes for MySQL / PostgreSQL / SQL Server are included as comments where
-- syntax differs.
-- =============================================================================

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    customer_id             TEXT PRIMARY KEY,
    state                   TEXT,
    account_length          INTEGER,
    area_code               INTEGER,
    international_plan      TEXT,       -- 'Yes' / 'No'
    voice_mail_plan         TEXT,       -- 'Yes' / 'No'
    number_vmail_messages   INTEGER,
    total_day_minutes       REAL,
    total_day_calls         INTEGER,
    total_day_charge        REAL,
    total_eve_minutes       REAL,
    total_eve_calls         INTEGER,
    total_eve_charge        REAL,
    total_night_minutes     REAL,
    total_night_calls       INTEGER,
    total_night_charge      REAL,
    total_intl_minutes      REAL,
    total_intl_calls        INTEGER,
    total_intl_charge       REAL,
    customer_service_calls  INTEGER,
    churn                   INTEGER,    -- 0/1  (SQLite has no native BOOLEAN)
    total_charge            REAL,
    total_minutes            REAL
);

-- MySQL/Postgres/SQL Server equivalent notes:
--   * SQLite REAL  -> MySQL/Postgres: DECIMAL(10,2) or FLOAT; SQL Server: FLOAT
--   * SQLite INTEGER for booleans -> Postgres: BOOLEAN; MySQL: TINYINT(1)
--   * TEXT PRIMARY KEY works the same in Postgres/MySQL (VARCHAR recommended
--     with an explicit length in MySQL, e.g. VARCHAR(20))

CREATE INDEX idx_customers_state ON customers(state);
CREATE INDEX idx_customers_churn ON customers(churn);
CREATE INDEX idx_customers_intl_plan ON customers(international_plan);
