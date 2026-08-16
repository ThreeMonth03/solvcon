# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Test deterministic projection and the point-cloud Atlas view."""

import math
import unittest
from unittest import mock

import solvcon

try:
    from PySide6 import QtCore, QtTest, QtWidgets
    from solvcon.pilot.panel import _matmul_benchmark_atlas as _atlas
    from solvcon.pilot.panel import _matmul_benchmark_cloud as _cloud
except ImportError:
    QtCore = QtTest = QtWidgets = None
    _atlas = _cloud = None


def _grouped_point():
    return {
        'x': 16.0,
        'y': 32.0,
        'z': 64.0,
        'route': '',
        'routes': ('blas_gemm', 'naive'),
        'winner_counts': {'blas_gemm': 1, 'naive': 1},
        'observation_ids': ('first', 'second'),
        'sample_count': 2,
        'hidden_sample_count': 1,
        'invalid_count': 0,
        'ambiguous_count': 1,
        'margin': 0.04,
        'noise': 0.12,
        'layouts': ('row-major / row-major',),
        'packing': ('eager: lhs', 'none'),
        'invalid': False,
        'has_invalid': False,
        'conflicting': True,
        'ambiguous': True,
    }


def _cloud_point(identifier, x_value, y_value, z_value):
    point = _grouped_point()
    point.update({
        'x': float(x_value),
        'y': float(y_value),
        'z': float(z_value),
        'route': 'naive',
        'routes': ('naive',),
        'winner_counts': {'naive': 1},
        'observation_ids': (identifier,),
        'sample_count': 1,
        'conflicting': False,
    })
    return point


class _FeatureRegistry:
    def names(self):
        return ['M', 'K', 'N']

    def evaluate(self, name, observation):
        return observation['contraction'][name.lower()]

    def register_expression(self, _name, _expression):
        raise AssertionError('This test does not register expressions')


def _observation(identifier, winner, n_value=64, k_value=32):
    return {
        'id': identifier,
        'winner': winner,
        'runner_up': None,
        'winner_margin': 0.08,
        'lhs': {'shape': [16, k_value], 'strides': [k_value, 1]},
        'rhs': {
            'shape': [k_value, n_value],
            'strides': [n_value, 1],
        },
        'contraction': {'m': 16, 'k': k_value, 'n': n_value},
        'routes': {
            winner: {
                'name': winner,
                'packing': {
                    'eager_lhs': winner == 'blas_gemm',
                    'eager_rhs': False,
                    'scratch_lhs': False,
                    'scratch_rhs': False,
                },
                'timing': {'median_ns': 400.0, 'mad_ns': 8.0},
            },
        },
    }


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class CloudProjectionTC(unittest.TestCase):
    def test_camera_presets_and_perspective_are_deterministic(self):
        camera = _cloud.CloudCamera()
        camera.set_view('front')
        front = camera.project_unit((1.0, 0.0, 0.0))
        self.assertAlmostEqual(front[0], 1.0)
        self.assertAlmostEqual(front[1], 0.0)

        camera.set_view('side')
        side = camera.project_unit((0.0, 0.0, 1.0))
        self.assertAlmostEqual(side[0], 1.0)
        self.assertAlmostEqual(side[1], 0.0)

        camera.set_view('top')
        top = camera.project_unit((0.0, 0.0, 1.0))
        self.assertAlmostEqual(top[0], 0.0)
        self.assertAlmostEqual(top[1], -1.0)

        camera.set_view('front')
        camera.perspective = True
        near = camera.project_unit((1.0, 0.0, 1.0))
        far = camera.project_unit((1.0, 0.0, -1.0))
        self.assertGreater(abs(near[0]), abs(far[0]))

        camera.reset()
        camera.orbit(20.0, -10.0)
        camera.zoom_by(1.0)
        self.assertAlmostEqual(camera.yaw, 47.0)
        self.assertAlmostEqual(camera.pitch, -28.5)
        self.assertGreater(camera.zoom, 1.0)

    def test_projection_preserves_exact_samples_and_depth_order(self):
        camera = _cloud.CloudCamera()
        camera.set_view('front')
        points = [
            {'x': 0.0, 'y': 5.0, 'z': 0.0, 'id': 'far'},
            {'x': 10.0, 'y': 5.0, 'z': 10.0, 'id': 'near'},
        ]

        projected = _cloud.project_points(
            points, camera, (0.0, 0.0, 200.0, 200.0))

        self.assertEqual(
            [item['point']['id'] for item in projected], ['far', 'near'])
        self.assertIs(projected[0]['point'], points[0])
        self.assertLess(
            projected[0]['screen'][0], projected[1]['screen'][0])
        self.assertEqual(projected[0]['normalized'], (-1.0, 0.0, -1.0))
        self.assertEqual(projected[1]['normalized'], (1.0, 0.0, 1.0))

    def test_hit_test_prefers_front_marker_when_points_overlap(self):
        back = {'screen': (50.0, 50.0), 'point': {'id': 'back'}}
        front = {'screen': (50.0, 50.0), 'point': {'id': 'front'}}

        selected = _cloud.hit_test(
            [back, front], (50.0, 50.0), radius=8.0)

        self.assertIs(selected, front)
        self.assertIsNone(
            _cloud.hit_test([back, front], (70.0, 70.0), radius=8.0))


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class CloudCanvasTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def test_canvas_renders_clusters_and_click_pins_exact_sample(self):
        canvas = _cloud.CloudCanvas()
        point = _grouped_point()
        canvas.set_points([point], 'M', 'K', 'N')
        canvas.resize(640, 420)

        image = canvas.grab().toImage()

        self.assertFalse(image.isNull())
        self.assertEqual(len(canvas._projected), 1)
        tooltip = canvas.point_tooltip(point)
        self.assertIn('M: 16', tooltip)
        self.assertIn('K: 32', tooltip)
        self.assertIn('N: 64', tooltip)
        self.assertIn('blas_gemm: 1', tooltip)
        self.assertIn('Minimum margin: 0.04', tooltip)
        self.assertIn('Maximum noise: 0.12', tooltip)
        self.assertIn('row-major / row-major', tooltip)
        self.assertIn('eager: lhs', tooltip)
        self.assertIn('first, second', tooltip)

        selected = []
        canvas.point_selected.connect(selected.append)
        center = canvas._projected[0]['screen']
        projection_generation = canvas._projection_generation
        QtTest.QTest.mouseClick(
            canvas, QtCore.Qt.MouseButton.LeftButton,
            pos=QtCore.QPoint(round(center[0]), round(center[1])))

        self.assertEqual(selected, [point])
        self.assertEqual(canvas._selected_key, canvas._point_key(point))
        self.assertEqual(
            canvas._projection_generation, projection_generation)

    def test_normalization_and_projection_caches_follow_state(self):
        canvas = _cloud.CloudCanvas()
        canvas.resize(640, 420)
        points = [
            _cloud_point('left', 0, 0, 0),
            _cloud_point('right', 10, 10, 10),
        ]
        with mock.patch.object(
                _cloud, '_normalized_values',
                wraps=_cloud._normalized_values) as normalize:
            canvas.set_points(points, 'M', 'K', 'N')
            self.assertEqual(normalize.call_count, 3)

            canvas._ensure_projected()
            generation = canvas._projection_generation
            canvas._ensure_projected()
            canvas.set_opacity(0.7)
            canvas.set_point_size(16)
            canvas._ensure_projected()
            self.assertEqual(canvas._projection_generation, generation)

            canvas.set_perspective(True)
            canvas._ensure_projected()
            self.assertEqual(
                canvas._projection_generation, generation + 1)
            canvas.set_view('side')
            canvas._ensure_projected()
            self.assertEqual(
                canvas._projection_generation, generation + 2)

            canvas._camera.orbit(8.0, 4.0)
            canvas._ensure_projected()
            canvas._camera.zoom_by(1.0)
            canvas._ensure_projected()
            canvas.resize(800, 500)
            canvas._ensure_projected()
            self.assertEqual(
                canvas._projection_generation, generation + 5)
            self.assertEqual(normalize.call_count, 3)

            canvas.set_points(points, 'M', 'K', 'N')
            canvas._ensure_projected()
            self.assertEqual(normalize.call_count, 6)
            self.assertEqual(
                canvas._projection_generation, generation + 6)

    def test_pick_reprojects_stale_camera_and_keeps_front_tie_break(self):
        canvas = _cloud.CloudCanvas()
        canvas.resize(640, 420)
        target = _cloud_point('target', 10, 0, 0)
        other = _cloud_point('other', 0, 10, 10)
        canvas.set_view('front')
        canvas.set_points([target, other], 'M', 'K', 'N')
        canvas._ensure_projected()
        old_screen = next(
            item['screen'] for item in canvas._projected
            if item['point'] is target)

        canvas.set_view('side')
        viewport = canvas._viewport(canvas._plot_rect())
        expected = _cloud.project_points(
            canvas._points, canvas._camera, viewport)
        new_screen = next(
            item['screen'] for item in expected if item['point'] is target)
        self.assertGreater(math.dist(old_screen, new_screen), 20.0)

        picked = canvas._pick_point(new_screen, radius=3.0)

        self.assertIs(picked['point'], target)
        actual_screen = next(
            item['screen'] for item in canvas._projected
            if item['point'] is target)
        self.assertEqual(actual_screen, new_screen)

        back = _cloud_point('back', 0, 0, 0)
        front = _cloud_point('front', 0, 0, 10)
        canvas.set_view('front')
        canvas.set_points([back, front], 'M', 'K', 'N')
        canvas._ensure_projected()
        center = canvas._projected[0]['screen']

        picked = canvas._pick_point(center, radius=3.0)

        self.assertIs(picked['point'], front)

    def test_spatial_index_matches_linear_pick_with_ten_thousand_points(self):
        canvas = _cloud.CloudCanvas()
        canvas.resize(1000, 700)
        canvas.set_view('front')
        points = [
            _cloud_point(str(row * 100 + column), column, row, 0)
            for row in range(100)
            for column in range(100)
        ]
        canvas.set_points(points, 'M', 'K', 'N')
        canvas._ensure_projected()
        position = canvas._projected[5050]['screen']

        indexed = canvas._pick_point(position, radius=8.0)
        linear = _cloud.hit_test(
            canvas._projected, position, radius=8.0)

        self.assertIs(indexed, linear)
        self.assertLess(canvas._last_pick_candidate_count, 1000)
        self.assertEqual(len(canvas._projected), 10000)

    def test_route_palette_is_stable_separated_and_matches_legend(self):
        routes = (
            'naive', 'blas_dot', 'blas_gevm', 'blas_gemv',
            'blas_gemm', 'winograd', 'numpy')
        first = [_cloud.route_color(route) for route in routes]
        second = [_cloud.route_color(route) for route in routes]

        self.assertEqual(
            [color.rgba() for color in first],
            [color.rgba() for color in second])
        for left_index, left in enumerate(first):
            for right in first[left_index + 1:]:
                distance = math.dist(
                    (left.red(), left.green(), left.blue()),
                    (right.red(), right.green(), right.blue()))
                self.assertGreaterEqual(distance, 100.0)

        points = []
        for index, route in enumerate(reversed(routes)):
            point = dict(_grouped_point())
            point.update({
                'x': float(index),
                'route': route,
                'routes': (route,),
                'winner_counts': {route: 1},
                'conflicting': False,
            })
            points.append(point)
        canvas = _cloud.CloudCanvas()
        canvas.set_points(points, 'M', 'K', 'N')

        self.assertEqual(canvas.legend_routes(), tuple(sorted(routes)))
        self.assertGreaterEqual(
            canvas._point_color('naive', 0.0).alphaF(), 0.75)


@unittest.skipUnless(solvcon.HAS_PILOT, 'Qt pilot is not built')
class CloudAtlasTC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance()
        if cls.app is None:
            cls.app = QtWidgets.QApplication([])

    def setUp(self):
        self.widget = _atlas.AtlasWidget(_FeatureRegistry())
        self.widget.add_artifacts([{
            'schema_kind': 'solvcon.matmul_benchmark',
            'request': {},
            'metadata': {},
            'observations': [
                _observation('first', 'blas_gemm'),
                _observation('second', 'naive'),
                _observation('third', 'naive', n_value=128),
            ],
        }])

    def test_atlas_only_projects_exact_3d_clusters(self):
        self.widget._x_axis.setCurrentText('M')
        self.widget._y_axis.setCurrentText('K')
        self.widget._z_axis.setCurrentText('N')
        self.widget.render()

        self.assertFalse(hasattr(self.widget, '_view'))
        self.assertFalse(hasattr(self.widget, '_canvas'))
        self.assertFalse(self.widget._cloud_controls.isHidden())
        self.assertEqual(len(self.widget._cloud_canvas._points), 2)
        first = self.widget._cloud_canvas._points[0]
        self.assertEqual((first['x'], first['y'], first['z']),
                         (16.0, 32.0, 64.0))
        self.assertEqual(first['sample_count'], 2)
        self.assertTrue(first['conflicting'])
        self.assertEqual(first['routes'], ('blas_gemm', 'naive'))
        self.assertIn('eager: lhs', first['packing'])

    def test_z_axis_ignores_matching_slice_without_hiding_volume(self):
        widget = _atlas.AtlasWidget(_FeatureRegistry())
        widget.add_artifacts([{
            'schema_kind': 'solvcon.matmul_benchmark',
            'request': {},
            'metadata': {},
            'observations': [
                _observation('k32', 'naive', k_value=32),
                _observation('k64', 'blas_gemm', k_value=64),
            ],
        }])
        widget._x_axis.setCurrentText('M')
        widget._y_axis.setCurrentText('N')
        widget._slice._feature.setCurrentText('K')
        widget._slice._enabled.setChecked(True)
        widget._slice._numeric.setValue(
            widget._slice._values.index(32))
        widget.render()

        self.assertEqual(
            sum(point['sample_count']
                for point in widget._cloud_canvas._points),
            1)

        widget._z_axis.setCurrentText('K')
        widget.render()

        self.assertEqual(
            sum(point['sample_count']
                for point in widget._cloud_canvas._points),
            2)
        self.assertIn('ignored', widget._slice.toolTip())
        self.assertIn('2 of 2 observations', widget._point_status.text())


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
