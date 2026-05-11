CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL,
    state TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    human_action TEXT NOT NULL,
    reward REAL NOT NULL,
    timestamp TEXT NOT NULL
);
