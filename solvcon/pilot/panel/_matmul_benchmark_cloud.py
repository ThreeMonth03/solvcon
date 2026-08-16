# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Interactive point-cloud rendering for matmul benchmark samples."""

import hashlib
import math

from PySide6 import QtCore, QtGui, QtWidgets

__all__ = [
    'CloudCamera',
    'CloudCanvas',
    'hit_test',
    'project_points',
    'route_color',
]

_ROUTE_COLORS = {
    'GenericIjk': '#0077bb',
    'DynamicIkj': '#66ccee',
    'BlasGemm': '#ee7733',
    'Strassen': '#228833',
    'Numpy': '#332288',
    'generic': '#0077bb',
    'blas_dot': '#f0e442',
    'blas_gevm': '#66ccee',
    'blas_gemv': '#aa3377',
    'blas_gemm': '#ee7733',
    'winograd': '#228833',
    'numpy': '#332288',
}

_PICK_BUCKET_SIZE = 32.0


def route_color(route):
    """Return a stable color for one benchmark route name."""
    if route in _ROUTE_COLORS:
        return QtGui.QColor(_ROUTE_COLORS[route])
    digest = hashlib.sha256(route.encode('utf8')).digest()
    hue = int.from_bytes(digest[:2], 'big') % 360
    return QtGui.QColor.fromHsv(hue, 225, 205)


class CloudCamera:
    """Hold a deterministic orbit camera for software projection."""

    _RESET_YAW = 38.0
    _RESET_PITCH = -24.0
    _PERSPECTIVE_DISTANCE = 4.0

    def __init__(self):
        self.perspective = False
        self.reset()

    def reset(self):
        self.yaw = self._RESET_YAW
        self.pitch = self._RESET_PITCH
        self.zoom = 1.0

    def set_view(self, name):
        views = {
            'front': (0.0, 0.0),
            'side': (90.0, 0.0),
            'top': (0.0, -90.0),
        }
        if name not in views:
            raise ValueError(f'Unknown cloud view: {name!r}')
        self.yaw, self.pitch = views[name]

    def orbit(self, horizontal_pixels, vertical_pixels):
        self.yaw = (self.yaw + 0.45 * horizontal_pixels) % 360.0
        self.pitch = max(
            -89.5, min(89.5, self.pitch + 0.45 * vertical_pixels))

    def zoom_by(self, wheel_steps):
        self.zoom = max(
            0.2, min(8.0, self.zoom * math.exp(0.14 * wheel_steps)))

    def project_unit(self, coordinates):
        """Project normalized XYZ coordinates into camera coordinates."""
        x_value, y_value, z_value = coordinates
        yaw = math.radians(self.yaw)
        pitch = math.radians(self.pitch)
        x_rotated = math.cos(yaw) * x_value + math.sin(yaw) * z_value
        z_rotated = -math.sin(yaw) * x_value + math.cos(yaw) * z_value
        y_rotated = (
            math.cos(pitch) * y_value - math.sin(pitch) * z_rotated)
        depth = math.sin(pitch) * y_value + math.cos(pitch) * z_rotated
        scale = 1.0
        if self.perspective:
            distance = self._PERSPECTIVE_DISTANCE
            scale = distance / max(0.5, distance - depth)
        return x_rotated * scale, -y_rotated * scale, depth


def _normalized_values(points, name):
    values = [float(point[name]) for point in points]
    low = min(values)
    high = max(values)
    if high == low:
        return [0.0] * len(values)
    extent = high - low
    return [2.0 * (value - low) / extent - 1.0 for value in values]


def _normalize_points(points):
    if not points:
        return ()
    x_values = _normalized_values(points, 'x')
    y_values = _normalized_values(points, 'y')
    z_values = _normalized_values(points, 'z')
    return tuple(
        (point, (x_value, y_value, z_value), index)
        for index, (point, x_value, y_value, z_value) in enumerate(zip(
            points, x_values, y_values, z_values)))


def _projection_scale(camera, viewport):
    _left, _top, width, height = viewport
    corners = (
        (x_value, y_value, z_value)
        for x_value in (-1.0, 1.0)
        for y_value in (-1.0, 1.0)
        for z_value in (-1.0, 1.0)
    )
    projected = [camera.project_unit(corner) for corner in corners]
    x_extent = max(abs(value[0]) for value in projected)
    y_extent = max(abs(value[1]) for value in projected)
    return 0.44 * min(
        width / max(0.1, x_extent),
        height / max(0.1, y_extent)) * camera.zoom


def _screen_position(projected, viewport, camera, scale=None):
    left, top, width, height = viewport
    center_x = left + width / 2.0
    center_y = top + height / 2.0
    if scale is None:
        scale = _projection_scale(camera, viewport)
    return center_x + projected[0] * scale, center_y + projected[1] * scale


def _project_normalized(points, camera, viewport, scale=None):
    if scale is None:
        scale = _projection_scale(camera, viewport)
    projected = []
    for point, normalized, input_index in points:
        camera_coordinates = camera.project_unit(normalized)
        projected.append({
            'point': point,
            'screen': _screen_position(
                camera_coordinates, viewport, camera, scale),
            'depth': camera_coordinates[2],
            'normalized': normalized,
            '_input_index': input_index,
        })
    projected.sort(key=lambda item: (item['depth'], item['_input_index']))
    return projected


def project_points(points, camera, viewport):
    """Project exact sample points and return them in painter depth order."""
    points = list(points)
    if not points:
        return []
    return _project_normalized(
        _normalize_points(points), camera, viewport)


def _hit_test_indices(projected, position, radius, indices):
    x_value, y_value = position
    maximum_distance = float(radius) ** 2
    best = None
    for painter_index in indices:
        item = projected[painter_index]
        x_distance = item['screen'][0] - x_value
        y_distance = item['screen'][1] - y_value
        distance = x_distance * x_distance + y_distance * y_distance
        if distance > maximum_distance:
            continue
        rank = (distance, -painter_index)
        if best is None or rank < best[0]:
            best = rank, item
    return None if best is None else best[1]


def hit_test(projected, position, radius):
    """Return the nearest marker, preferring the front marker on a tie."""
    return _hit_test_indices(
        projected, position, radius, range(len(projected)))


def _build_screen_index(projected, bucket_size):
    buckets = {}
    for painter_index, item in enumerate(projected):
        x_value, y_value = item['screen']
        if not math.isfinite(x_value) or not math.isfinite(y_value):
            continue
        key = (
            math.floor(x_value / bucket_size),
            math.floor(y_value / bucket_size),
        )
        buckets.setdefault(key, []).append(painter_index)
    return {key: tuple(indices) for key, indices in buckets.items()}


def _nearby_indices(index, position, radius, bucket_size):
    x_value, y_value = position
    radius = abs(float(radius))
    x_start = math.floor((x_value - radius) / bucket_size)
    x_stop = math.floor((x_value + radius) / bucket_size)
    y_start = math.floor((y_value - radius) / bucket_size)
    y_stop = math.floor((y_value + radius) / bucket_size)
    indices = []
    for x_bucket in range(x_start, x_stop + 1):
        for y_bucket in range(y_start, y_stop + 1):
            indices.extend(index.get((x_bucket, y_bucket), ()))
    return indices


class CloudCanvas(QtWidgets.QWidget):
    """Render exact benchmark samples as an interactive 3D point cloud."""

    point_selected = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._camera = CloudCamera()
        self._points = []
        self._normalized_points = ()
        self._data_generation = 0
        self._projected = []
        self._projection_key = None
        self._screen_scale = None
        self._projection_generation = 0
        self._screen_index = {}
        self._last_pick_candidate_count = 0
        self._axis_names = ('', '', '')
        self._point_size = 12.0
        self._opacity = 0.9
        self._last_position = None
        self._dragged = False
        self._selected_key = None
        self.setMouseTracking(True)
        self.setMinimumSize(420, 300)

    def set_points(self, points, x_name, y_name, z_name):
        self._points = list(points)
        self._normalized_points = _normalize_points(self._points)
        self._data_generation += 1
        self._invalidate_projection()
        self._axis_names = x_name, y_name, z_name
        keys = {self._point_key(point) for point in self._points}
        if self._selected_key not in keys:
            self._selected_key = None
        self.update()

    def set_perspective(self, enabled):
        enabled = bool(enabled)
        if enabled != self._camera.perspective:
            self._camera.perspective = enabled
            self._invalidate_projection()
            self.update()

    def set_point_size(self, size):
        self._point_size = max(4.0, min(36.0, float(size)))
        self.update()

    def set_opacity(self, opacity):
        self._opacity = max(0.1, min(1.0, float(opacity)))
        self.update()

    def reset_view(self):
        previous = self._camera_state()
        perspective = self._camera.perspective
        self._camera.reset()
        self._camera.perspective = perspective
        if self._camera_state() != previous:
            self._invalidate_projection()
            self.update()

    def set_view(self, name):
        previous = self._camera_state()
        self._camera.set_view(name)
        if self._camera_state() != previous:
            self._invalidate_projection()
            self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        plot = self._plot_rect()
        viewport = self._viewport(plot)
        self._ensure_projected(viewport)
        self._draw_frame(painter, plot)
        if not self._points:
            painter.drawText(
                plot, QtCore.Qt.AlignmentFlag.AlignCenter,
                'Load benchmark results to render this view')
            return

        self._draw_axes(painter, viewport)
        for item in self._projected:
            self._draw_point(painter, item)
        self._draw_legend(painter, plot)

    def _plot_rect(self):
        return QtCore.QRectF(
            22, 28, max(1, self.width() - 44),
            max(1, self.height() - 52))

    @staticmethod
    def _viewport(plot):
        return plot.left(), plot.top(), plot.width(), plot.height()

    def _camera_state(self):
        return (
            self._camera.yaw,
            self._camera.pitch,
            self._camera.zoom,
            self._camera.perspective,
        )

    def _invalidate_projection(self):
        self._projection_key = None

    def _ensure_projected(self, viewport=None):
        if viewport is None:
            viewport = self._viewport(self._plot_rect())
        key = self._data_generation, self._camera_state(), viewport
        if key == self._projection_key:
            return

        self._screen_scale = _projection_scale(
            self._camera, viewport)
        self._projected = _project_normalized(
            self._normalized_points, self._camera, viewport,
            self._screen_scale)
        self._screen_index = _build_screen_index(
            self._projected, _PICK_BUCKET_SIZE)
        self._projection_key = key
        self._projection_generation += 1

    def _pick_point(self, position, radius=None):
        self._ensure_projected()
        if radius is None:
            radius = self._point_size / 2.0 + 4.0
        indices = _nearby_indices(
            self._screen_index, position, radius, _PICK_BUCKET_SIZE)
        self._last_pick_candidate_count = len(indices)
        return _hit_test_indices(
            self._projected, position, radius, indices)

    def _draw_frame(self, painter, plot):
        painter.setPen(self.palette().mid().color())
        painter.drawRect(plot)
        projection = (
            'perspective' if self._camera.perspective else 'orthographic')
        painter.drawText(
            QtCore.QRectF(plot.left(), 4, plot.width(), 20),
            QtCore.Qt.AlignmentFlag.AlignLeft, projection)

    def _draw_axes(self, painter, viewport):
        origin = (-1.0, -1.0, -1.0)
        endpoints = (
            (1.0, -1.0, -1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, -1.0, 1.0),
        )
        origin_position = self._project_unit(
            origin, viewport, self._screen_scale)
        colors = ('#d62728', '#2ca02c', '#1f77b4')
        for name, endpoint, color in zip(
                self._axis_names, endpoints, colors):
            position = self._project_unit(
                endpoint, viewport, self._screen_scale)
            pen = QtGui.QPen(QtGui.QColor(color))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(
                QtCore.QPointF(*origin_position), QtCore.QPointF(*position))
            painter.drawText(
                QtCore.QPointF(position[0] + 3, position[1] - 3), name)

    def _project_unit(self, coordinates, viewport, scale=None):
        camera_coordinates = self._camera.project_unit(coordinates)
        return _screen_position(
            camera_coordinates, viewport, self._camera, scale)

    def _draw_point(self, painter, item):
        point = item['point']
        center = QtCore.QPointF(*item['screen'])
        size = self._point_size
        marker = QtCore.QRectF(
            center.x() - size / 2.0, center.y() - size / 2.0,
            size, size)
        painter.save()
        if point['invalid']:
            painter.setBrush(QtGui.QBrush(
                QtGui.QColor(105, 105, 105, 150),
                QtCore.Qt.BrushStyle.Dense4Pattern))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(marker)
        elif point['conflicting']:
            self._draw_conflicting_routes(painter, marker, point)
        else:
            color = self._point_color(point['route'], point['margin'])
            painter.setBrush(color)
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(marker)

        pen = QtGui.QPen(self.palette().text().color())
        pen.setWidth(2 if point['ambiguous'] or point['noise'] > 0.1 else 1)
        if point['ambiguous']:
            pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        elif point['noise'] > 0.1:
            pen.setStyle(QtCore.Qt.PenStyle.DotLine)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        painter.drawEllipse(marker)
        if point['has_invalid']:
            painter.drawLine(marker.topLeft(), marker.bottomRight())
            painter.drawLine(marker.topRight(), marker.bottomLeft())
        if self._point_key(point) == self._selected_key:
            selected = marker.adjusted(-4, -4, 4, 4)
            selected_pen = QtGui.QPen(QtGui.QColor('#e15759'))
            selected_pen.setWidth(3)
            painter.setPen(selected_pen)
            painter.drawEllipse(selected)
        if not self._dragged and point['sample_count'] > 1 and size >= 12:
            painter.setPen(self.palette().text().color())
            painter.drawText(
                marker, QtCore.Qt.AlignmentFlag.AlignCenter,
                str(point['sample_count']))
        painter.restore()

    def _point_color(self, route, margin):
        color = route_color(route)
        confidence = max(0.0, min(1.0, float(margin)))
        color.setAlphaF(self._opacity * (0.85 + 0.15 * confidence))
        return color

    def _draw_conflicting_routes(self, painter, marker, point):
        routes = point['routes']
        span = 360 * 16 // len(routes)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        for index, route in enumerate(routes):
            painter.setBrush(self._point_color(route, point['margin']))
            painter.drawPie(marker, index * span, span)

    def _draw_legend(self, painter, plot):
        x_position = plot.right() - 8
        y_position = plot.top() + 8
        painter.setPen(self.palette().text().color())
        for route in self.legend_routes():
            width = painter.fontMetrics().horizontalAdvance(route) + 24
            x_position -= width
            painter.setBrush(route_color(route))
            painter.drawEllipse(
                QtCore.QRectF(x_position, y_position, 12, 12))
            painter.drawText(
                QtCore.QPointF(x_position + 16, y_position + 11), route)
            x_position -= 8

    def legend_routes(self):
        """Return the stable route order represented by legend swatches."""
        return tuple(sorted(set(
            route for point in self._points for route in point['routes'])))

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._last_position = event.position()
            self._dragged = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._last_position is not None and event.buttons() \
                & QtCore.Qt.MouseButton.LeftButton:
            difference = event.position() - self._last_position
            if abs(difference.x()) + abs(difference.y()) >= 1.0:
                self._dragged = True
                self._camera.orbit(difference.x(), difference.y())
                self._invalidate_projection()
                self._last_position = event.position()
                self.update()
            self.setToolTip('')
            QtWidgets.QToolTip.hideText()
            return
        item = self._pick_point(
            (event.position().x(), event.position().y()))
        if item is None:
            self.setToolTip('')
            QtWidgets.QToolTip.hideText()
            return
        tooltip = self.point_tooltip(item['point'])
        self.setToolTip(tooltip)
        QtWidgets.QToolTip.showText(
            event.globalPosition().toPoint(), tooltip, self)

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if not self._dragged:
            item = self._pick_point(
                (event.position().x(), event.position().y()))
            self._selected_key = (
                None if item is None else self._point_key(item['point']))
            if item is not None:
                self.point_selected.emit(item['point'])
            self.update()
        self._last_position = None
        self._dragged = False
        self.update()
        event.accept()

    def wheelEvent(self, event):
        steps = event.angleDelta().y() / 120.0
        if steps:
            self._camera.zoom_by(steps)
            self._invalidate_projection()
            self.update()
        event.accept()

    def resizeEvent(self, event):
        self._invalidate_projection()
        super().resizeEvent(event)

    def leaveEvent(self, event):
        self.setToolTip('')
        QtWidgets.QToolTip.hideText()
        super().leaveEvent(event)

    def point_tooltip(self, point):
        counts = ', '.join(
            f'{route}: {count}'
            for route, count in sorted(point['winner_counts'].items()))
        identifiers = [
            str(identifier) for identifier in point['observation_ids']
            if identifier is not None
        ]
        shown_ids = ', '.join(identifiers[:8])
        if len(identifiers) > 8:
            shown_ids += f', ... ({len(identifiers)} total)'
        layouts = ', '.join(point.get('layouts', ())) or 'unknown'
        packing = ', '.join(point.get('packing', ())) or 'none'
        x_name, y_name, z_name = self._axis_names
        return '\n'.join((
            f'{x_name}: {point["x"]}',
            f'{y_name}: {point["y"]}',
            f'{z_name}: {point["z"]}',
            f'Samples: {point["sample_count"]}',
            f'Winners: {counts or "none"}',
            f'Minimum margin: {point["margin"]}',
            f'Maximum noise: {point["noise"]}',
            f'Invalid: {point["invalid_count"]}',
            'Mixed or near-tie results: '
            f'{"yes" if point["ambiguous"] else "no"}',
            f'Layouts: {layouts}',
            f'Packing: {packing}',
            f'IDs: {shown_ids or "none"}',
        ))

    @staticmethod
    def _point_key(point):
        return (
            point['x'], point['y'], point['z'],
            tuple(point['observation_ids']))


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
