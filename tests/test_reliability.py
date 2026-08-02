import unittest

from code.reliability import RetryPolicy, run_with_retry


class ReliabilityTests(unittest.TestCase):
    def test_transient_error_retries_with_exponential_delay(self):
        attempts, delays = [], []
        policy = RetryPolicy(3, "exponential", 0.5, 4.0, 0.0, 30.0)

        def operation(attempt):
            attempts.append(attempt)
            if attempt < 2:
                raise TimeoutError("temporary timeout")
            return "ok"

        self.assertEqual("ok", run_with_retry(operation, policy, sleeper=delays.append,
                                               random_source=lambda: 0.0))
        self.assertEqual([0, 1, 2], attempts)
        self.assertEqual([0.5, 1.0], delays)

    def test_terminal_error_is_not_retried(self):
        attempts = []
        policy = RetryPolicy(3, "exponential", 0.5, 4.0, 0.0, 30.0)

        def operation(attempt):
            attempts.append(attempt)
            raise RuntimeError("authentication failed")

        with self.assertRaises(RuntimeError):
            run_with_retry(operation, policy, sleeper=lambda _delay: None)
        self.assertEqual([0], attempts)

    def test_none_mode_disables_retries(self):
        attempts = []
        policy = RetryPolicy(3, "none", 0.5, 4.0, 0.0, 30.0)

        def operation(attempt):
            attempts.append(attempt)
            raise TimeoutError("temporary timeout")

        with self.assertRaises(TimeoutError):
            run_with_retry(operation, policy, sleeper=lambda _delay: None)
        self.assertEqual([0], attempts)


if __name__ == "__main__":
    unittest.main()
