"""Read-only, rate-limited per-frame vision diagnostics. No Maix dependency."""

import app_config as config
from target_selector import (
    expanded_target_neighbors,
    directional_target_neighbors,
    casualty_vertical_grab_allowed,
    selected_target_rectangles,
    selected_target_x_padding,
    spacing_neighbor_in_y_scope,
)
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
    wire_id = packet[2]
    tagged = bool(wire_id & 0x80)
    details = "id={} dx={} dy={}".format(
        wire_id & 0x0F if tagged else wire_id,
        -x if packet[5] == 2 else x, -y if packet[8] == 2 else y
    )
    if tagged:
        details += " area_gate={}".format(
            "open" if wire_id & 0x20 else "near" if wire_id & 0x40 else "far"
        )
    return details


class VisionDiagnostics:
    """Explain raw/verified/eligible/selected stages without changing them."""

    def __init__(self):
        self._last_ms = None
        self._last_signature = None
        self._last_frame = 0
        self._last_compact_key = None

    def report(self, cam, labels, raw, verified, eligible, selected,
               verification, before_mode, policy, before_anchor,
               previous_position, jump_rejected, continuity_removed, now_ms,
               anchor_attribute, overlap_objects=None):
        if not config.VISION_DIAGNOSTICS_ENABLED:
            return
        verbose = config.VISION_DIAGNOSTICS_VERBOSE
        with cam._lock:
            frame = cam._latest_submit_sequence
            mode = cam.mode
            target = cam.target_label
            compact_key = (mode, target)
            if (not verbose and compact_key == self._last_compact_key
                    and self._last_ms is not None
                    and now_ms - self._last_ms < config.VISION_DIAGNOSTICS_INTERVAL_MS):
                # Return before building unused snapshots/command lists on
                # every frame. Detection flicker cannot bypass this limit.
                return
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
            zone_area = cam._zone_area_ratio
            zone_missing = cam._zone_no_target_phase
            zone_realign = cam._zone_realign_phase
        signature = (before_mode, mode, target, selected is not None,
                     int(selected.class_id) if selected is not None else None,
                     flags, anchor is not None)
        confirmed_boundary = (track_missing == config.UART_TRACK_LOST_CONFIRM_FRAMES
                              or search_missing == config.UART_SEARCH_NO_TARGET_FRAMES
                              or final_missing == config.UART_FINAL_NO_TARGET_FRAMES)
        if (verbose and signature == self._last_signature and not confirmed_boundary
                and self._last_ms is not None
                and now_ms - self._last_ms < config.VISION_DIAGNOSTICS_INTERVAL_MS):
            return
        delta_ms = 0 if self._last_ms is None else now_ms - self._last_ms
        fps = (frame - self._last_frame) * 1000.0 / delta_ms if delta_ms > 0 else 0.0
        self._last_signature, self._last_ms, self._last_frame = signature, now_ms, frame
        self._last_compact_key = compact_key
        prefix = "t={} f={}".format(now_ms, frame)
        def emit(tag, message):
            print("[{}] {} {}".format(tag, prefix, message))
        if not verbose:
            selected_id = "id{}".format(int(selected.class_id)) if selected is not None else "none"
            zone_info = ""
            if before_mode == 0x06:
                zone_info = " area={:.1%} missingE6={} wait36={} realign46={} sent26={}".format(
                    zone_area, zone_missing, flags[2], zone_realign, flags[4])
            emit("VISION", "state={:X} target={} selected={} raw={} verified={} fps={:.1f} miss03/04/24={}/{}/{} wait44={} waitF1={} coord_tx={}{}".format(
                mode, target, selected_id, len(raw), len(verified), fps,
                search_missing, track_missing, final_missing, flags[0], flags[1], tx_count, zone_info))
            return
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
        if policy in ("arrange02_center_lock", "arrange02_center_acquire"):
            emit("ARRANGE02-TRACK", "source={} reference=image_center lock_before={} selected={} side_selection=no coordinates={} no_auto34 wait_MCU12".format(
                "current_raw_yolo" if policy == "arrange02_center_lock" else "verified_new_target3",
                before_anchor, object_name(selected) if selected is not None else "none",
                "fresh" if selected is not None else "paused"))
        if policy == "final24_yolo_fallback":
            emit("FINAL24-FALLBACK", "source=current_raw_yolo nearest=roi_center_2d merged_applied=no verification_applied=no selected={}".format(
                object_name(selected) if selected is not None else "none"))
        if policy in ("search_lock", "track_lock", "final_lock", "search_hold", "track_hold"):
            lock_class = before_anchor[0] if before_anchor is not None else None
            lock_raw = [obj for obj in raw
                        if (int(obj.class_id) in (5, 6)
                            if cam._uses_relaxed_supply_tracking(target)
                            else int(obj.class_id) == lock_class)]
            lock_outside = cam._exclude_objects_in_safety_zones(lock_raw, labels, raw)
            emit("LOCK-FOLLOW", "source=current_raw_yolo merged_applied=no verification_applied=no selected={} loss_frames={} previous={} raw_same_class={} outside_zone={}".format(
                object_name(selected) if selected is not None else "none",
                config.UART_TRACK_LOST_CONFIRM_FRAMES if before_mode != 0x24 else 1,
                before_anchor, len(lock_raw), len(lock_outside)))
        eligible_ids = {id(obj) for obj in eligible}
        excluded = [obj for obj in verified if id(obj) not in eligible_ids]
        if before_mode in (0x03, 0x04, 0x104, 0x114, 0x102):
            roi_excluded = [obj for obj in excluded
                            if cam._target_center_in_tracking_roi(obj)]
            if roi_excluded:
                roi_tag = ("SEARCH-ROI-EXCLUDE" if before_mode in (0x03, 0x104)
                           else "ARRANGE-ROI-EXCLUDE" if before_mode == 0x102
                           else "TRACK-ROI-EXCLUDE")
                emit(roi_tag, "center_in_roi=[{}] roi={}".format(
                    object_list(roi_excluded), cam.tracking_roi))
                excluded = [obj for obj in excluded
                            if not cam._target_center_in_tracking_roi(obj)]
        emit("SAFE-EXCLUDE", object_list(excluded))
        if excluded:
            zones = [obj for obj in raw if int(obj.class_id) in (1, 2)]
            safety_margin_x = 0 if before_mode == 0x06 else config.UART_SAFE_ZONE_TARGET_MARGIN_X
            zone_bounds = [
                (zone, cam._safety_zone_exclusion_bounds(zone, safety_margin_x))
                for zone in zones if float(zone.w) > 0 and float(zone.h) > 0
            ]
            for obj in excluded[:limit]:
                x, y = float(obj.x) + float(obj.w) * .5, float(obj.y) + float(obj.h) * .5
                covering = [zone for zone, (left, top, right, bottom) in zone_bounds
                            if left <= x <= right and top <= y <= bottom]
                emit("SAFE-REASON", "{} covered_by=[{}] x_padding={}".format(
                    object_name(obj), object_list(covering), safety_margin_x))
        if continuity_removed:
            emit("CONTINUITY", "removed=[{}] previous={} limits=x{} y{} area{:.0%}".format(
                object_list(continuity_removed), previous_position, config.UART_MAX_X_JUMP_PX,
                config.UART_MAX_Y_JUMP_PX, config.UART_MAX_AREA_CHANGE_RATIO))
        neighbor_source = eligible
        if policy in ("first_green_overlap14", "supply_overlap14", "red_overlap14"):
            neighbor_source = (overlap_objects if overlap_objects is not None
                               else cam._exclude_objects_in_safety_zones(raw, labels, raw))
            if overlap_objects is None and cam._uses_supply_collection():
                neighbor_source = cam._unloaded_supply_candidates(neighbor_source)
        movable = [obj for obj in neighbor_source if 0 < int(obj.class_id) < len(labels)
                   and labels[int(obj.class_id)] in ("sqareblue", "sqarered", "sqareredgreen", "triangualrblack")]
        target_labels = (("sqareredgreen", "triangualrblack")
                         if cam._uses_relaxed_supply_tracking(target) or policy == "supply_overlap14"
                         else cam._eligible_target_labels(target))
        candidates = [obj for obj in movable if labels[int(obj.class_id)] in target_labels]
        for obj in candidates[:limit]:
            x, y = float(obj.x) + float(obj.w) * .5, float(obj.y) + float(obj.h) * .5
            if policy in ("first_green_overlap14", "supply_overlap14", "red_overlap14"):
                _, expanded = selected_target_rectangles(obj, cam.frame_width, cam.frame_height)
                neighbor_objects = cam._directional_neighbor_objects(obj, movable)
                ignored_roi = [other for other in movable if other is not obj
                               and cam._target_center_in_tracking_roi(other)]
                ignored_supplies = [other for other in movable
                                    if other not in neighbor_objects and other not in ignored_roi]
                neighbors = expanded_target_neighbors(obj, neighbor_objects, cam.frame_width, cam.frame_height)
                horizontal, vertical = directional_target_neighbors(obj, neighbor_objects, cam.frame_width, cam.frame_height)
                corners = [other for other in neighbors if other not in horizontal and other not in vertical]
                padding = selected_target_x_padding(y)
                emit("CANDIDATE", "{} rule=expanded_box_overlap rect={} x_padding={} y_padding={} horizontal=[{}] vertical=[{}] corner_ignored=[{}] red_vertical_grab_ignore={} ignored_supplies=[{}] roi_ignored=[{}] merged_applied=no".format(
                    object_name(obj), expanded, padding, padding, object_list(horizontal), object_list(vertical), object_list(corners),
                    casualty_vertical_grab_allowed(obj, horizontal, vertical), object_list(ignored_supplies), object_list(ignored_roi)))
                continue
            neighbors, front = [], []
            for other in movable:
                if other is obj:
                    continue
                ox, oy = float(other.x) + float(other.w) * .5, float(other.y) + float(other.h) * .5
                threshold, _ = cam._final_spacing_limits(y, int(obj.class_id) == 4 or int(other.class_id) == 4)
                if spacing_neighbor_in_y_scope(
                    obj.class_id, y, oy, cam.frame_height * 0.5
                ) and abs(ox - x) <= threshold:
                    neighbors.append("r{}:id{} dx={:.1f}<={}".format(raw_indices.get(id(other), "?"), other.class_id, abs(ox-x), threshold))
                if ((before_mode != 0x24 or policy == "supply_no_spacing")
                        and min(obj.x + obj.w, other.x + other.w) > max(obj.x, other.x) and oy > y):
                    front.append("r{}:id{} y={:.1f}".format(raw_indices.get(id(other), "?"), other.class_id, oy))
            if policy in ("supply_no_spacing", "supply_no_spacing24"):
                verdict = "skip"
            else:
                verdict = "reject" if neighbors or front else "pass"
            emit("CANDIDATE", "{} merged={} applied={} close=[{}] front=[{}] roi_y_error={:.1f} front_applied={}".format(
                object_name(obj), verdict,
                "yes" if policy in ("merged", "final24_spacing") else "no",
                ";".join(neighbors) or "none", ";".join(front) or "none", abs(y-cam._search_selection_reference()[1]),
                "yes" if policy in ("merged", "supply_no_spacing") else "no"))
        required = (zone_label,) if policy == "zone" else target_labels
        def contains_target(items):
            return any(0 < int(obj.class_id) < len(labels) and labels[int(obj.class_id)] in required for obj in items)
        if selected is not None:
            reason = "selected"
        elif policy == "near05":
            reason = "count_only"
        elif policy == "new_target_wait_outside":
            reason = "NEW_TARGET_CONFIRMING_OUTSIDE_ZONE"
        elif policy in ("first_green_overlap14", "supply_overlap14", "red_overlap14"):
            reason = "OVERLAP_NEIGHBOR" if candidates else "YOLO_NO_TARGET"
        elif policy in ("search_hold", "track_hold"):
            reason = "LOCK_TARGET_IN_SAFE_ZONE" if lock_raw and not lock_outside else "LOCK_CLASS_MISSING_HOLD"
        elif not contains_target(raw):
            reason = "YOLO_NO_TARGET"
        elif policy == "zone" and zone_area <= getattr(config, "UART_ZONE_MIN_AREA_RATIO", .02):
            reason = "ZONE_AREA_TOO_SMALL"
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
            emit("NEAR-DETAIL", "ids={} action_bucket={} stable={}/{} settle_remaining_ms={} waiting05tx={}".format(
                cam._near_view_target_ids(verified, labels), near_bucket, near_stable,
                config.UART_NEAR_VIEW_STABLE_FRAMES,
                (config.UART_NEAR_VIEW_SETTLE_MS if near_started is None else
                 max(0, config.UART_NEAR_VIEW_SETTLE_MS-(now_ms-near_started))),
                near_started is None))
        if policy == "zone":
            polygon = zone_obstacle_polygon(cam.tracking_roi, cam.frame_width, cam.frame_height)
            blockers = [obj for obj in movable if point_in_zone_obstacle_polygon(
                float(obj.x)+float(obj.w)*.5, float(obj.y)+float(obj.h)*.5, polygon)]
            area = zone_area
            ratio = config.UART_ZONE_CASUALTY_X_RATIO if target == config.UART_ZONE_CASUALTY_LABEL else config.UART_ZONE_SUPPLY_X_RATIO
            if flags[3] and area > config.UART_ZONE_CENTER_AREA_RATIO:
                ratio = (config.UART_ZONE_CASUALTY_NEAR_X_RATIO
                         if target == config.UART_ZONE_CASUALTY_LABEL
                         else config.UART_ZONE_SUPPLY_NEAR_X_RATIO)
            emit("ZONE-DETAIL", "selected={} area={:.3f} selection=largest_visible_area min_area>{:.2f} missingE6={} center_switch>{:.2f} close>={:.2f} aim_x_ratio={} polygon={} blockers=[{}] realign46={}".format(
                object_name(selected) if selected is not None else "none", area,
                getattr(config, "UART_ZONE_MIN_AREA_RATIO", .02), zone_missing,
                config.UART_ZONE_CENTER_AREA_RATIO, config.UART_ZONE_CLOSE_AREA_RATIO, ratio,
                polygon, object_list(blockers), zone_realign))
