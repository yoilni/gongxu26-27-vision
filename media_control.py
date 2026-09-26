"""Streaming and video-recording lifecycle management."""

import gc
import os
from datetime import datetime

from maix import audio, err, image, rtsp, sys, time, video

import app_config as config


class MediaController:
    def __init__(self, main_camera):
        self.main_camera = main_camera
        self.active_protocol = None
        self.stream_camera = None
        self.stream_protocol = None
        self.stream_server = None

        self.record_audio_enabled = True
        self.record_camera = None
        self.record_recorder = None
        self.record_microphone = None
        self.record_path = ""

    @property
    def recording(self):
        return self.record_recorder is not None

    def toggle_audio_option(self):
        if self.recording:
            print("Stop recording before changing the audio option")
            return
        self.record_audio_enabled = not self.record_audio_enabled
        print(
            "Recording audio:",
            "enabled" if self.record_audio_enabled else "disabled",
        )

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
            return

        if self.active_protocol is not None:
            old_protocol = self.active_protocol
            self.stop_stream()
            self.main_camera.clear_buff()
            print(old_protocol, "streaming stopped for recording")
        self.start_recording()

    def toggle_stream(self, protocol):
        if protocol == config.PROTOCOL_RTSP and not config.RTSP_DYNAMIC_ENABLED:
            print(
                "RTSP is disabled: native Rtsp.stop() crashes on this firmware; "
                "please use WebRTC"
            )
            return

        if self.recording:
            self.stop_recording()

        if self.active_protocol == protocol:
            self.stop_stream()
            self.main_camera.clear_buff()
            print(protocol, "streaming stopped")
            return

        if self.active_protocol is not None:
            old_protocol = self.active_protocol
            self.stop_stream()
            self.main_camera.clear_buff()
            print(old_protocol, "streaming stopped")

        self.start_stream(protocol)

    def start_stream(self, protocol):
        if protocol not in (
            config.PROTOCOL_RTSP,
            config.PROTOCOL_WEBRTC,
        ):
            print("Unsupported stream protocol:", protocol)
            return

        server = None
        try:
            if self.stream_camera is None:
                self.stream_camera = self.main_camera.add_channel(
                    config.STREAM_WIDTH,
                    config.STREAM_HEIGHT,
                    image.Format.FMT_YVU420SP,
                    fps=-1,
                    buff_num=config.STREAM_BUFFER_NUM,
                )
                self.stream_protocol = protocol
                print(protocol, "camera channel created")
            elif self.stream_protocol != protocol:
                raise RuntimeError(
                    "retained stream channel belongs to "
                    + str(self.stream_protocol)
                )

            if protocol == config.PROTOCOL_RTSP:
                server = rtsp.Rtsp(
                    fps=config.STREAM_FPS,
                    bitrate=config.STREAM_BITRATE,
                )
            elif protocol == config.PROTOCOL_WEBRTC:
                from maix import webrtc

                server = webrtc.WebRTC()
            else:
                raise ValueError("unsupported stream protocol: " + str(protocol))

            err.check_raise(
                server.bind_camera(self.stream_camera),
                protocol + " camera bind failed",
            )
            err.check_raise(server.start(), protocol + " start failed")
            self.stream_server = server
            self.active_protocol = protocol
            print(protocol, "streaming started")
            self._print_stream_client_urls(protocol, server)
        except Exception as exc:
            print(protocol, "start failed:", exc)
            if server is not None:
                try:
                    server.stop()
                except Exception:
                    pass
            server = None
            self.stream_server = None
            self.active_protocol = None
            gc.collect()
            if self.stream_camera is not None:
                print(protocol, "camera channel retained after start failure")

    def stop_stream(self, destroy=False):
        protocol = self.active_protocol or self.stream_protocol or "STREAM"
        if (
            protocol == config.PROTOCOL_RTSP
            and self.stream_server is not None
        ):
            print("RTSP stop skipped: native Rtsp.stop() is unsafe on this firmware")
            return

        server = self.stream_server
        if server is not None:
            try:
                print(protocol, "stop stage 1/4: stopping server")
                result = server.stop()
                if result != err.Err.ERR_NONE:
                    print(protocol, "stop failed:", err.to_str(result))
                print(protocol, "stop stage 2/4: server stopped")
            except Exception as exc:
                print(protocol, "stop failed:", exc)

        self.stream_server = None
        server = None
        gc.collect()
        print(protocol, "stop stage 3/4: server resources released")

        if self.stream_camera is not None:
            if protocol == config.PROTOCOL_WEBRTC and not destroy:
                print(
                    protocol,
                    "stop stage 4/4: camera channel retained for safe reuse",
                )
            else:
                print(protocol, "stop stage 4/4: closing camera channel")
                self.stream_camera.close()
                self.stream_camera = None
                self.stream_protocol = None
                print(protocol, "camera channel closed")
        self.active_protocol = None

    def _make_record_path(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(
            config.RECORD_DIR,
            "record_{}.mp4".format(timestamp),
        )
        if not os.path.exists(base_path):
            return base_path

        index = 1
        while True:
            candidate = os.path.join(
                config.RECORD_DIR,
                "record_{}_{:02d}.mp4".format(timestamp, index),
            )
            if not os.path.exists(candidate):
                return candidate
            index += 1

    def start_recording(self):
        record_camera = None
        recorder = None
        microphone = None
        output_path = self._make_record_path()
        try:
            record_camera = self.main_camera.add_channel(
                config.RECORD_WIDTH,
                config.RECORD_HEIGHT,
                image.Format.FMT_YVU420SP,
                fps=config.RECORD_FPS,
                buff_num=config.RECORD_BUFFER_NUM,
            )
            if self.record_audio_enabled:
                microphone = audio.Recorder(
                    path="",
                    sample_rate=config.RECORD_AUDIO_SAMPLE_RATE,
                    format=audio.Format.FMT_S16_LE,
                    channel=config.RECORD_AUDIO_CHANNELS,
                    block=True,
                )

            recorder = video.VideoRecorder()
            err.check_raise(
                recorder.bind_camera(record_camera),
                "record camera bind failed",
            )
            if microphone is not None:
                err.check_raise(
                    recorder.bind_audio(microphone),
                    "record microphone bind failed",
                )
            err.check_raise(recorder.reset(), "record recorder reset failed")
            if microphone is not None:
                recorder.mute(False)
                recorder.volume(config.RECORD_AUDIO_VOLUME)
            err.check_raise(
                recorder.config_path(output_path),
                "record path config failed",
            )
            err.check_raise(
                recorder.config_bitrate(config.RECORD_BITRATE),
                "record bitrate config failed",
            )
            err.check_raise(
                recorder.config_fps(config.RECORD_FPS),
                "record fps config failed",
            )

            if microphone is not None:
                microphone.reset(True)
                time.sleep_ms(120)
            err.check_raise(recorder.record_start(), "record start failed")

            self.record_camera = record_camera
            self.record_recorder = recorder
            self.record_microphone = microphone
            self.record_path = output_path
            if microphone is not None:
                print(
                    "Recording started with audio: {} ({} Hz, mono)".format(
                        output_path,
                        config.RECORD_AUDIO_SAMPLE_RATE,
                    )
                )
            else:
                print("Recording started without audio:", output_path)
        except Exception as exc:
            print("Recording start failed:", exc)
            if recorder is not None:
                try:
                    recorder.record_finish()
                    recorder.close()
                except Exception as cleanup_exc:
                    print("Recorder cleanup failed:", cleanup_exc)
            if microphone is not None:
                try:
                    microphone.reset(False)
                    microphone.finish()
                except Exception as cleanup_exc:
                    print("Microphone cleanup failed:", cleanup_exc)
            gc.collect()
            if record_camera is not None:
                record_camera.close()

    def stop_recording(self):
        had_audio = self.record_microphone is not None
        recorder = self.record_recorder
        if recorder is not None:
            try:
                err.check_raise(recorder.record_finish(), "record finish failed")
                recorder.close()
                if self.record_path:
                    print(
                        "Recording saved {} audio: {}".format(
                            "with" if had_audio else "without",
                            self.record_path,
                        )
                    )
            except Exception as exc:
                print("Failed to finalize recording:", exc)
        self.record_recorder = None
        recorder = None
        gc.collect()

        microphone = self.record_microphone
        if microphone is not None:
            try:
                microphone.reset(False)
                microphone.finish()
            except Exception as exc:
                print("Failed to close microphone:", exc)
        self.record_microphone = None
        microphone = None
        gc.collect()

        if self.record_camera is not None:
            self.record_camera.close()
        self.record_camera = None
        self.record_path = ""

    @staticmethod
    def _usable_network_addresses():
        addresses = []
        for interface, ip_address in sys.ip_address().items():
            if (
                ip_address
                and ip_address != "0.0.0.0"
                and not ip_address.startswith("127.")
            ):
                addresses.append((interface, ip_address))
        return addresses

    def _print_stream_client_urls(self, protocol, server):
        addresses = self._usable_network_addresses()
        if not addresses:
            print(protocol, "started, but no usable network IP was found")
            print("Connect WiFi/USB network first, then restart streaming")
            return

        if protocol == config.PROTOCOL_RTSP:
            print("Open one of these URLs in VLC:")
            for interface, ip_address in addresses:
                print("  {}: rtsp://{}:8554/live".format(interface, ip_address))
            return

        try:
            raw_urls = list(server.get_urls())
        except Exception:
            raw_urls = []
        if not raw_urls:
            try:
                raw_url = server.get_url()
            except Exception as exc:
                print("Failed to get WebRTC URL:", exc)
                raw_url = ""
            if raw_url:
                raw_urls = [raw_url]

        print("Open one of these URLs in Chrome:")
        for raw_url in raw_urls:
            if "0.0.0.0" in raw_url:
                for interface, ip_address in addresses:
                    print(
                        "  {}: {}".format(
                            interface,
                            raw_url.replace("0.0.0.0", ip_address),
                        )
                    )
            else:
                print(" ", raw_url)

    def close(self):
        self.stop_recording()
        self.stop_stream(destroy=True)
