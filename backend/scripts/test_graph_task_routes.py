"""Exercise the task-list HTTP route with the real TaskManager serialization."""
import unittest

from flask import Flask

from app.api import graph_bp
from app.models.task import TaskManager, TaskStatus


class GraphTaskRoutesTest(unittest.TestCase):
    def setUp(self):
        self.manager = TaskManager()
        self.previous_tasks = self.manager._tasks
        self.manager._tasks = {}
        app = Flask(__name__)
        app.register_blueprint(graph_bp, url_prefix="/api/graph")
        self.client = app.test_client()

    def tearDown(self):
        self.manager._tasks = self.previous_tasks

    def test_empty_task_list(self):
        response = self.client.get("/api/graph/tasks")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"success": True, "data": [], "count": 0})

    def test_pending_and_terminal_tasks_keep_their_serialized_state(self):
        pending = self.manager.create_task("graph_build")
        failed = self.manager.create_task("graph_build")
        self.manager.update_task(failed, status=TaskStatus.FAILED, error="fixture error")
        response = self.client.get("/api/graph/tasks")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["count"], 2)
        rows = {row["task_id"]: row for row in response.json["data"]}
        self.assertEqual(rows[pending]["status"], "pending")
        self.assertEqual(rows[failed]["status"], "failed")
        self.assertEqual(rows[failed]["error"], "fixture error")
        detail = self.client.get(f"/api/graph/task/{failed}")
        self.assertEqual(rows[failed], detail.json["data"])


if __name__ == "__main__":
    unittest.main()
