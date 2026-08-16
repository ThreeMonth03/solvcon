# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""Atomic artifact persistence and side-effect-free dataset merging."""

import datetime
import json
import os
import pathlib
import tempfile
import uuid

from . import schema


def write_artifact(artifact, path):
    """Atomically replace path with one validated benchmark document."""

    schema.validate_document(artifact)
    path = pathlib.Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf8', dir=path.parent,
                prefix=f'.{path.name}.', suffix='.tmp',
                delete=False) as stream:
            temporary_path = pathlib.Path(stream.name)
            json.dump(artifact, stream, indent=2, sort_keys=True,
                      allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def load_artifact(path):
    """Load one validated artifact or merged collection."""

    with pathlib.Path(path).open(encoding='utf8') as stream:
        artifact = json.load(stream)
    return schema.validate_document(artifact)


def merge_artifacts(artifacts):
    """Merge existing artifacts without starting a benchmark collector."""

    loaded = []
    sources = []
    panels = []
    observations = []
    for source_index, item in enumerate(artifacts):
        if isinstance(item, (str, os.PathLike)):
            artifact = schema.validate_artifact(load_artifact(item))
            source_path = str(pathlib.Path(item).expanduser().resolve())
        else:
            artifact = schema.validate_artifact(item)
            source_path = None
        loaded.append(artifact)
        source_id = f'source-{source_index}'
        sources.append({
            'source_id': source_id,
            'artifact_id': artifact['artifact_id'],
            'path': source_path,
            'created_at': artifact['created_at'],
            'request': artifact['request'],
            'metadata': artifact['metadata'],
            'panel_count': len(artifact['panels']),
            'panels_sha256': schema.panels_sha256(artifact['panels']),
        })
        panels.extend({
            'source_id': source_id,
            'source_artifact_id': artifact['artifact_id'],
            'source_panel_index': panel_index,
            'panel': panel,
        } for panel_index, panel in enumerate(artifact['panels']))
        observations.extend({
            'source_id': source_id,
            'source_observation_index': observation_index,
            'observation': observation,
        } for observation_index, observation in enumerate(
            artifact['observations']))
    collection = {
        'schema_version': schema.SCHEMA_VERSION,
        'schema_kind': schema.COLLECTION_KIND,
        'collection_id': uuid.uuid4().hex,
        'created_at': datetime.datetime.now(datetime.UTC).isoformat(),
        'sources': sources,
        'panels': panels,
        'observations': observations,
        'artifact_count': len(loaded),
    }
    return schema.validate_document(collection)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
