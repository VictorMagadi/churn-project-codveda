-- =============================================================================
-- Level 2 - Task 2: SQL for Business Analytics — query library
-- Codveda Business Analytics Internship
-- Run against the `customers` table created by 01_create_tables_sqlite.sql
-- and populated by sql/run_sql_demo.py (or load churn_cleaned.csv yourself
-- into any RDBMS and point these queries at it).
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Basic aggregation: overall churn KPIs
-- -----------------------------------------------------------------------------
SELECT
    COUNT(*)                                   AS total_customers,
    SUM(churn)                                  AS total_churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2)     AS churn_rate_pct,
    ROUND(AVG(total_charge), 2)                 AS avg_monthly_charge,
    ROUND(AVG(customer_service_calls), 2)       AS avg_service_calls
FROM customers;

-- -----------------------------------------------------------------------------
-- 2. GROUP BY + aggregation: churn rate and usage by state
-- -----------------------------------------------------------------------------
SELECT
    state,
    COUNT(*)                                    AS n_customers,
    SUM(churn)                                   AS n_churned,
    ROUND(100.0 * SUM(churn) / COUNT(*), 2)      AS churn_rate_pct,
    ROUND(AVG(total_day_minutes), 2)             AS avg_day_minutes,
    ROUND(AVG(total_charge), 2)                  AS avg_total_charge
FROM customers
GROUP BY state
ORDER BY churn_rate_pct DESC
LIMIT 10;

-- -----------------------------------------------------------------------------
-- 3. Aggregation by plan type (international + voicemail combos)
-- -----------------------------------------------------------------------------
SELECT
    international_plan,
    voice_mail_plan,
    COUNT(*)                                     AS n_customers,
    ROUND(100.0 * AVG(churn), 2)                 AS churn_rate_pct,
    ROUND(AVG(total_charge), 2)                  AS avg_total_charge
FROM customers
GROUP BY international_plan, voice_mail_plan
ORDER BY churn_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- 4. HAVING clause: states with churn rate above the company average
-- -----------------------------------------------------------------------------
SELECT
    state,
    COUNT(*)                                      AS n_customers,
    ROUND(100.0 * AVG(churn), 2)                  AS churn_rate_pct
FROM customers
GROUP BY state
HAVING AVG(churn) > (SELECT AVG(churn) FROM customers)
ORDER BY churn_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- 5. Self-join style comparison via a subquery: customers whose day-minute
--    usage is above their own state's average (usage outliers within state)
-- -----------------------------------------------------------------------------
SELECT
    c.customer_id, c.state, c.total_day_minutes, s.state_avg_minutes
FROM customers c
JOIN (
    SELECT state, AVG(total_day_minutes) AS state_avg_minutes
    FROM customers
    GROUP BY state
) s ON c.state = s.state
WHERE c.total_day_minutes > s.state_avg_minutes * 1.5
ORDER BY c.total_day_minutes DESC
LIMIT 20;

-- -----------------------------------------------------------------------------
-- 6. Window functions: rank customers by total charge within each state
--    (SQLite >= 3.25, MySQL >= 8.0, Postgres, SQL Server all support this
--    standard window-function syntax unchanged)
-- -----------------------------------------------------------------------------
SELECT
    customer_id, state, total_charge,
    RANK() OVER (PARTITION BY state ORDER BY total_charge DESC) AS charge_rank_in_state,
    ROUND(AVG(total_charge) OVER (PARTITION BY state), 2)        AS state_avg_charge
FROM customers
ORDER BY state, charge_rank_in_state
LIMIT 30;

-- -----------------------------------------------------------------------------
-- 7. Running total / cumulative distribution of customer service calls
--    (window function ORDER BY without PARTITION = whole-table running sum)
-- -----------------------------------------------------------------------------
SELECT
    customer_service_calls,
    COUNT(*) AS n_customers,
    SUM(COUNT(*)) OVER (ORDER BY customer_service_calls) AS running_total_customers
FROM customers
GROUP BY customer_service_calls
ORDER BY customer_service_calls;

-- -----------------------------------------------------------------------------
-- 8. CASE WHEN: bucket customers into usage tiers directly in SQL
-- -----------------------------------------------------------------------------
SELECT
    CASE
        WHEN total_day_minutes < 100 THEN 'Low usage'
        WHEN total_day_minutes < 200 THEN 'Medium usage'
        ELSE 'High usage'
    END AS usage_tier,
    COUNT(*)                                AS n_customers,
    ROUND(100.0 * AVG(churn), 2)            AS churn_rate_pct
FROM customers
GROUP BY usage_tier
ORDER BY churn_rate_pct DESC;

-- -----------------------------------------------------------------------------
-- 9. Query optimization example: same question as (2) but written to use the
--    index on `state` and avoid a full scan for the churn filter — compare
--    EXPLAIN QUERY PLAN output before/after adding idx_customers_churn
-- -----------------------------------------------------------------------------
EXPLAIN QUERY PLAN
SELECT state, COUNT(*) FROM customers WHERE churn = 1 GROUP BY state;

-- -----------------------------------------------------------------------------
-- 10. Top customer-service-call offenders per state (window + filter pattern,
--     a common "top-N per group" business question)
-- -----------------------------------------------------------------------------
WITH ranked AS (
    SELECT
        customer_id, state, customer_service_calls, churn,
        ROW_NUMBER() OVER (PARTITION BY state ORDER BY customer_service_calls DESC) AS rn
    FROM customers
)
SELECT * FROM ranked WHERE rn <= 3
ORDER BY state, rn;
