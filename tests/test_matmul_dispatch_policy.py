# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest
from unittest import mock

from contrib.matmul_dispatch import environment


class MatmulDispatchPolicyTC(unittest.TestCase):

    def test_tuning_sources_match_current_checkout(self):
        sources = {"contrib/tune_matmul_dispatch.py": "current"}
        records = ({
            "sample_id": "matching",
            "environment": {"tuning_source_sha256": sources},
        },)
        with mock.patch.object(
                environment, "_source_hashes", return_value=sources):
            environment.validate_tuning_sources(records)


if __name__ == "__main__":
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
