class WebRTCService:
    def handle_offer(self, sdp: str, sdp_type: str) -> dict:
        return {"sdp": sdp, "type": "answer"}

    def handle_candidate(self, candidate: str, sdp_mid: str | None, sdp_mline_index: int | None) -> dict:
        return {"ok": True}

    def handle_interrupt(self) -> dict:
        return {"ok": True}
