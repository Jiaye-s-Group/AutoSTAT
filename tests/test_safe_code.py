from __future__ import annotations

import json
import os
import pickle
import unittest
from unittest.mock import patch

from sklearn.ensemble import RandomForestRegressor

from core.safe_code import (
    UnsafeCodeError,
    generated_code_execution_policy,
    restricted_pickle_loads,
    safe_exec,
    safe_subprocess_env,
    validate_code,
)


class _MaliciousPickle:
    def __reduce__(self):
        return (os.system, ("echo should-not-run",))


class SafeCodeTests(unittest.TestCase):
    def test_safe_exec_allows_typical_dataframe_code(self) -> None:
        namespace: dict[str, object] = {}
        safe_exec(
            """
import pandas as pd
values = pd.Series([1, 2, 2]).replace(2, 3)
result_dict = {"total": int(values.sum())}
""",
            namespace,
        )
        self.assertEqual(namespace["result_dict"], {"total": 7})

    def test_generated_code_policy_exposes_enforced_restrictions(self) -> None:
        from core.code_runtime_profile import build_code_runtime_constraints

        policy = generated_code_execution_policy()
        self.assertEqual(policy["executor"], "safe_exec")
        self.assertIn("getattr", policy["forbidden_names"])
        self.assertIn("hasattr", policy["allowed_builtin_names"])

        constraints = json.loads(build_code_runtime_constraints([], target=""))
        self.assertEqual(constraints["execution_safety"], policy)

    def test_rejects_environment_and_file_access(self) -> None:
        unsafe_samples = (
            "import os\nresult = os.environ",
            "result = open('/etc/passwd').read()",
            "result = pd.read_csv('/tmp/data.csv')",
            "result = globals()",
            "result = value.__class__",
        )
        for code in unsafe_samples:
            with self.subTest(code=code):
                with self.assertRaises(UnsafeCodeError):
                    validate_code(code)

    def test_allows_model_class_without_dunder_introspection(self) -> None:
        namespace: dict[str, object] = {}
        safe_exec(
            """
class Scaler:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, value):
        return value * self.factor

result = Scaler(3)(4)
""",
            namespace,
        )
        self.assertEqual(namespace["result"], 12)

    def test_subprocess_environment_omits_application_secrets(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "secret",
                "AUTH_DB_URL": "secret-db",
                "PATH": "safe-path",
            },
            clear=True,
        ):
            env = safe_subprocess_env()
        self.assertEqual(env["PATH"], "safe-path")
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("AUTH_DB_URL", env)

    def test_restricted_unpickler_accepts_plain_data(self) -> None:
        value = {"models": ["linear"], "score": 0.9}
        self.assertEqual(restricted_pickle_loads(pickle.dumps(value)), value)

    def test_restricted_unpickler_accepts_trained_sklearn_model(self) -> None:
        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit([[0], [1], [2]], [0, 1, 2])
        restored = restricted_pickle_loads(pickle.dumps(model))
        self.assertEqual(restored.predict([[1]]).shape, (1,))

    def test_restricted_unpickler_rejects_command_globals(self) -> None:
        payload = pickle.dumps(_MaliciousPickle())
        with self.assertRaises(pickle.UnpicklingError):
            restricted_pickle_loads(payload)


if __name__ == "__main__":
    unittest.main()
