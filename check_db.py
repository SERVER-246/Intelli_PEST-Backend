"""Quick script to check database contents."""
import sqlite3

conn = sqlite3.connect('D:/Intelli_PEST-Backend/feedback_data/intellipest.db')
cursor = conn.cursor()

# List all tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]
print("=" * 60)
print("DATABASE: intellipest.db")
print("=" * 60)
print(f"\nTABLES: {tables}\n")

# Count records in each table
for table in tables:
    cursor.execute(f"SELECT COUNT(*) FROM {table}")
    count = cursor.fetchone()[0]
    print(f"  {table}: {count} records")

print("\n" + "=" * 60)
print("DETAILED DATA:")
print("=" * 60)

# Show users
print("\n--- USERS ---")
cursor.execute("SELECT user_id, email, total_submissions, total_feedbacks, trust_score FROM users")
for row in cursor.fetchall():
    print(f"  {row}")

# Show recent training runs
print("\n--- TRAINING RUNS ---")
cursor.execute("SELECT run_id, training_type, status, model_version_string, kd_enabled, created_at FROM training_runs ORDER BY created_at DESC LIMIT 5")
for row in cursor.fetchall():
    print(f"  {row}")

# Show recent system events
print("\n--- RECENT SYSTEM EVENTS ---")
cursor.execute("SELECT event_type, component, message, created_at FROM system_events ORDER BY created_at DESC LIMIT 5")
for row in cursor.fetchall():
    print(f"  {row}")

conn.close()
print("\n✅ Database is PERSISTENT - works without server running!")
