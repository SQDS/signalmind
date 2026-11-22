"""Additional answer formats registered without changing historical prompt bytes."""

AUDIT_ANSWER_FORMATS = {
    "audit_rate": '{"bpm": <mean pulse rate in beats per minute>}',
    "audit_motion": '{"t": <burst start in seconds from the supplied window start>}',
    "audit_step": '{"t": <step start in seconds from the supplied window start>}',
    "audit_unusable": '{"percent": <unusable percentage of the supplied window, 0-100>}',
    "audit_bp": '{"sbp": <mean SBP in mmHg>, "dbp": <mean DBP in mmHg>, "map": <mean MAP in mmHg>}',
    "audit_valsalva": '{"t": <onset in seconds from the supplied window start>}',
    "audit_damped": '{"t": <stretch start in seconds from the supplied window start>}',
    "audit_tilt": '{"t": <onset in seconds from the supplied window start>}',
    "audit_shift": '{"channel": <exact channel name>, "shift_s": <signed imposed shift in seconds>}',
    "audit_pat": '{"pat_ms": <mean pulse arrival time in milliseconds>}',
    "audit_corrupted": '{"channel": <exact corrupted channel name>}',
    "audit_response": (
        '{"hr_direction": <"increase"|"decrease"|"unchanged">, '
        '"sbp_direction": <"increase"|"decrease"|"unchanged">, '
        '"hr_latency_s": <seconds after tilt onset, or null if unchanged>, '
        '"sbp_latency_s": <seconds after tilt onset, or null if unchanged>}'
    ),
    "audit_brs": '{"brs_ms_mmhg": <sequence baroreflex sensitivity in ms/mmHg>}',
    "audit_cbfv_tilt": '{"t": <onset in seconds from the supplied window start>}',
    "audit_tfa": '{"gain": <LF transfer-function gain in (cm/s)/mmHg>}',
}
