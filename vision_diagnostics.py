"""Read-only, rate-limited per-frame vision diagnostics. No Maix dependency."""

import app_config as config
from zone_obstacle_region import zone_obstacle_polygon, point_in_zone_obstacle_polygon


def describe_object(obj):
    if obj is None:
        return "none"
    return "id{}@({:.1f},{:.1f}) wh={:.1f}x{:.1f} area={:.0f} score={:.3f}".format(
        int(obj.class_id), float(obj.x) + float(obj.w) * 0.5,
        float(obj.y) + float(obj.h) * 0.5, float(obj.w), float(obj.h),
        float(obj.w) * float(obj.h), float(getattr(obj, "score", 0.0)),
    )


def coordinate_details(packet):
    if packet is None or len(packet) != 11:
        return "none"
    x = (packet[3] << 8) | packet[4]
    y = (packet[6] << 8) | packet[7]
    return "id={} dx={} dy={}".format(
        packet[2], -x if packet[5] == 2 else x, -y if packet[8] == 2 else y
    )


class VisionDiagnostics:
    """Explain raw/verified/eligible/selected stages without changing them."""

    def __init__(self):
        self._last_ms = None
        self._last_signature = None
        self._last_frame = 0

    def report(self, cam, labels, raw, verified, eligible, selected,
               verification, before_mode, policy, before_anchor,
               previous_position, jump_rejected, continuity_removed, now_ms,
               anchor_attribute):
        if not config.VISION_DIAGNOSTICS_ENABLED:
            return
        with cam._lock:
            frame = cam._latest_submit_sequence
            mode = cam.mode
            target = cam.target_label
            zone_label = cam.zone_label
            track_missing = cam._track_no_target_frames
            search_missing = cam._search_no_target_frames
            final_missing = cam._final_no_target_frames
            anchor = getattr(cam, anchor_attribute)
            packet = cam._latest_packet
            pending = [event[2] for event in cam._pending_events]
            flags = (cam._track_green_recovery_waiting, cam._final_search_waiting_ack,
                     cam._zone_obstacle_waiting, cam._zone_found_sent, cam._zone_close_sent,
                     cam._final_roi_sent, cam._track_no_target_sent)
            obstacle_hits = cam._zone_obstacle_frames
            tx_count = cam._tx_coordinate_count
            tx_ms = cam._last_coordinate_write_ms
            near_started = cam._near_view_started_ms
            near_stable = cam._near_view_stable_frames
            near_bucket = cam._near_view_stable_bucket
        signature = (before_mode, mode, target, selected is not None,
                     int(selected.class_id) if selected is not None else None,
                     flags, anchor is not None)
        confirmed_boundary = (track_missing == config.UART_TRACK_LOST_CONFIRM_FRAMES
                              or search_missing == config.UART_SEARCH_NO_TARGET_FRAMES
                              or final_missing == config.UART_FINAL_NO_TARGET_FRAMES)
        if (signature == self._last_signature and not confirmed_boundary
                and self._last_ms is not None
                and now_ms - self._last_ms < config.VISION_DIAGNOSTICS_INTERVAL_MS):
            return
        delta_ms = 0 if self._last_ms is None else now_ms - self._last_ms
        fps = (frame - self._last_frame) * 1000.0 / delta_ms if delta_ms > 0 else 0.0
        self._last_signature, self._last_ms, self._last_frame = signature, now_ms, frame
        prefix = "t={} f={}".format(now_ms, frame)
        def emit(tag, message):
            print("[{}] {} {}".format(tag, prefix, message))
        limit = max(1, int(config.VISION_DIAGNOSTICS_MAX_OBJECTS))
        raw_indices = {id(obj): index for index, obj in enumerate(raw)}
        def object_name(obj):
            return "r{}:{}".format(raw_indices.get(id(obj), "?"), describe_object(obj))
        def object_list(items):
            text = " | ".join(object_name(obj) for obj in items[:limit]) or "none"
            if len(items) > limit:
                text += " | omitted={}".format(len(items) - limit)
            return text
        emit("FRAME", "state={:X}->{:X} target={} zone={} policy={} fps={:.1f} raw={} verified={} eligible={}".format(
            before_mode, mode, target, zone_label, policy, fps, len(raw), len(verified), len(eligible)))
        emit("RAW", object_list(raw))
        if verification:
            for record in verification[:limit]:
                emit("VOBJ", "{} status={} reason={} H={}/{} E={} evidence={} high_conf_bypass={} confirmed={} roi={} aspect={:.2f} R={:.3f} G={:.3f} B={:.3f} K={:.3f} blob={:.3f} density={:.3f}".format(
                    object_name(record["obj"]), record.get("status", "unknown"),
                    record.get("reason", "n/a"), record.get("hits", 0), config.VERIFY_REQUIRED_HITS,
                    record.get("evidence", 0), record.get("current_evidence", False),
                    record.get("high_conf_bypass", False),
                    record.get("confirmed", False), record.get("roi"), record.get("aspect", 0),
                    record.get("red", 0), record.get("green", 0), record.get("blue", 0),
                    record.get("black", 0), record.get("largest_blob_ratio", 0), record.get("density", 0)))
        emit("VERIFIED", object_list(verified))
        if policy in ("search_lock", "track_lock", "final_lock", "search_hold", "track_hold"):
            lock_class = before_anchor[0] if before_anchor is not None else None
            lock_raw = [obj for obj in raw if int(obj.class_id) == lock_class]
            lock_outside = cam._exclude_objects_in_safety_zones(lock_raw, labels, raw)
            emit("LOCK-FOLLOW", "source=current_raw_yolo merged_applied=no verification_applied=no selected={} loss_frames={} previous={} raw_same_class={} outside_zone={}".format(
                object_name(selected) if selected is not None else "none",
                config.UART_TRACK_LOST_CONFIRM_FRAMES if before_mode != 0x24 else 1,
                before_anchor, len(lock_raw), len(lock_outside)))
        eligible_ids = {id(obj) for obj in eligible}
        excluded = [obj for obj in verified if id(obj) not in eligible_ids]
        emit("SAFE-EXCLUDE", object_list(excluded))
        if excluded:
            zones = [obj for obj in raw if int(obj.class_id) in (1, 2)]
            for obj in excluded[:limit]:
                x, y = float(obj.x) + float(obj.w) * .5, float(obj.y) + float(obj.h) * .5
                covering = [zone for zone in zones
                            if zone.x <= x <= zone.x + zone.w and zone.y <= y <= zone.y + zone.h]
                emit("SAFE-REASON", "{} covered_by=[{}]".format(object_name(obj), object_list(covering)))
        if continuity_removed:
            emit("CONTINUITY", "removed=[{}] previous={} limits=x{} y{} area{:.0%}".format(
                object_list(continuity_removed), previous_position, config.UART_MAX_X_JUMP_PX,
                config.UART_MAX_Y_JUMP_PX, config.UART_MAX_AREA_CHANGE_RATIO))
        movable = [obj for obj in eligible if 0 < int(obj.class_id) < len(labels)
                   and labels[int(obj.class_id)] in ("sqareblue", "sqarered", "sqareredgreen", "triangualrblack")]
        candidates = [obj for obj in movable if labels[int(obj.class_id)] in cam._eligible_target_labels(target)]
        for obj in candidates[:limit]:
            x, y = float(obj.x) + float(obj.w) * .5, float(obj.y) + float(obj.h) * .5
            neighbors, front = [], []
            for other in movable:
                if other is obj:
                    continue
                ox, oy = float(other.x) + float(other.w) * .5, float(other.y) + float(other.h) * .5
                threshold, _ = cam._final_spacing_limits(y, int(obj.class_id) == 4 or int(other.class_id) == 4)
                if abs(ox - x) <= threshold:
                    neighbors.append("r{}:id{} dx={:.1f}<={}".format(raw_indices.get(id(other), "?"), other.class_id, abs(ox-x), threshold))
                if min(obj.x + obj.w, other.x + other.w) > max(obj.x, other.x) and oy > y:
                    front.append("r{}:id{} y={:.1f}".format(raw_indices.get(id(other), "?"), other.class_id, oy))
            emit("CANDIDATE", "{} merged={} applied={} close=[{}] front=[{}] roi_y_error={:.1f}".format(
                object_name(obj), "reject" if neighbors or front else "pass",
                "yes" if policy == "merged" else "no",
                ";".join(neighbors) or "none", ";".join(front) or "none", abs(y-cam._search_selection_reference()[1])))
        required = (zone_label,) if policy == "zone" else cam._eligible_target_labels(target)
        def contains_target(items):
            return any(0 < int(obj.class_id) < len(labels) and labels[int(obj.class_id)] in required for obj in items)
        if selected is not None:
            reason = "selected"
        elif policy == "near05":
            reason = "count_only"
        elif policy in ("search_hold", "track_hold"):
            reason = "LOCK_TARGET_IN_SAFE_ZONE" if lock_raw and not lock_outside else "LOCK_CLASS_MISSING_HOLD"
        elif not contains_target(raw):
            reason = "YOLO_NO_TARGET"
        elif policy != "zone" and not contains_target(verified):
            reason = "VERIFICATION_REJECTED"
        elif policy != "zone" and not contains_target(eligible):
            reason = "SAFE_ZONE_EXCLUDED"
        elif jump_rejected:
            reason = "JUMP_REJECTED"
        elif before_anchor is not None and anchor is not None:
            reason = "LOCK_CLASS_MISSING_HOLD"
        elif continuity_removed and not contains_target([
                obj for obj in eligible if all(obj is not removed for removed in continuity_removed)]):
            reason = "CONTINUITY_REJECTED"
        else:
            reason = "NO_ELIGIBLE_SELECTION"
        emit("SELECT", "result={} reason={} lock_before={} lock_after={} reference={} prepared={} pending={} missing03={}/{} missing04={}/{} missing24={}/{}".format(
            object_name(selected) if selected is not None else "none", reason, before_anchor, anchor,
            cam._reference_point(before_mode), coordinate_details(packet), ["{:02X}".format(v) for v in pending],
            search_missing, config.UART_SEARCH_NO_TARGET_FRAMES, track_missing, config.UART_TRACK_LOST_CONFIRM_FRAMES,
            final_missing, config.UART_FINAL_NO_TARGET_FRAMES))
        emit("WAIT", "green44={} EE_F1={} zone36={} zone16={} zone26={} roi34={} E4_sent={} obstacle_hits={}/{} actual_coord_tx={} last_tx_age_ms={}".format(
            *flags, obstacle_hits, config.UART_ZONE_OBSTACLE_CONFIRM_FRAMES, tx_count,
            now_ms-tx_ms if tx_ms is not None else "never"))
        if policy == "near05":
            emit("NEAR-DETAIL", "ids={} action_bucket={} stable={}/{} settle_remaining_ms={}".format(
                cam._movable_target_ids(verified, labels), near_bucket, near_stable,
                config.UART_NEAR_VIEW_STABLE_FRAMES,
                max(0, config.UART_NEAR_VIEW_SETTLE_MS-(now_ms-near_started))))
        if policy == "zone":
            polygon = zone_obstacle_polygon(cam.tracking_roi, cam.frame_width, cam.frame_height)
            blockers = [obj for obj in movable if point_in_zone_obstacle_polygon(
                float(obj.x)+float(obj.w)*.5, float(obj.y)+float(obj.h)*.5, polygon)]
            area = cam._visible_box_area_ratio(selected) if selected is not None else 0
            ratio = config.UART_ZONE_CASUALTY_X_RATIO if target == config.UART_ZONE_CASUALTY_LABEL else config.UART_ZONE_SUPPLY_X_RATIO
            if flags[3] and area > config.UART_ZONE_CENTER_AREA_RATIO:
                ratio = .5
            emit("ZONE-DETAIL", "selected={} area={:.3f} center_switch>{:.2f} close>={:.2f} aim_x_ratio={} polygon={} blockers=[{}]".format(
                object_name(selected) if selected is not None else "none", area,
                config.UART_ZONE_CENTER_AREA_RATIO, config.UART_ZONE_CLOSE_AREA_RATIO, ratio,
                polygon, object_list(blockers)))
