"""Deterministic request packing. An over-budget item stays pending; members never silently disappear."""

import json
from pathlib import Path

from PIL import Image

from tsmcp.inspection.profiles import profile


def encoded(obj):
    return len(json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


class ContextBuilder:
    def __init__(self, config):
        budgets = profile(config.get("inspection_preset", "balanced"))["budgets"]
        self.calls = config.get("max_llm_calls_per_record", budgets["max_llm_calls_per_record"])
        self.images = config.get("max_images_per_call", budgets["max_images_per_call"])
        self.text = config.get("max_text_bytes", 65536)
        self.tokens = config.get("max_request_tokens", 131072)
        self.output = config.get("max_output_tokens", 4096)
        self.seconds = config.get("max_request_seconds", 120)
        if (
            any(type(v) is not int or v <= 0 for v in (self.images, self.text, self.tokens, self.seconds, self.output))
            or type(self.calls) is not int
            or self.calls < 0
            or self.output >= self.tokens
        ):
            raise ValueError("Invalid request budgets")

    def group_pages(self, request):
        """Partition only the members; keep both representatives, explicit global indices and case mapping."""
        base = request["data"]
        rows = base["member_rows"]
        n = len(rows)
        # Leave space for the system, schema, budget accounting and per-role context.
        picture_reserve = self.usage([dict(request, data={})], "", {})["reserved_tokens"]
        limit = min(self.text, self.tokens - picture_reserve)
        if encoded(base) + 8192 <= limit or n == 1:
            return [(request, 0, n)]

        def page(a, b):
            data = dict(base)
            subset = rows[a:b]
            cases = sorted({row[0] for row in subset})
            index = {j: i for i, j in enumerate(cases)}
            data.update(
                member_offset=a,
                group_size=n,
                visual_group_id=request["id"],
                scope=b - a,
                cases=[base["cases"][j] for j in cases],
                member_rows=[[index[row[0]], *row[1:]] for row in subset],
                image_scope="global group representatives; image_rows index the original group",
                representative_members=[
                    dict(case=base["cases"][rows[i][0]], values=rows[i][1:]) for i in base["image_rows"]
                ],
            )
            if "core_samples_by_member" in base:
                data["core_samples_by_member"] = base["core_samples_by_member"][a:b]
            return dict(request, id=f"{request['id']}-members-{a}-{b}", data=data)

        pages = []
        a = 0
        while a < n:
            lo, hi = a + 1, n
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if encoded(page(a, mid)["data"]) + 8192 <= limit:
                    lo = mid
                else:
                    hi = mid - 1
            pages.append((page(a, lo), a, lo))
            a = lo
        return pages

    def usage(self, rows, system, schema):
        images = [p for r in rows for p in r["images"]]
        text = encoded([system, schema, *[dict(id=r["id"], data=r["data"]) for r in rows]])
        # Conservative local token units; provider-reported token counts are kept separately.
        visual = 0
        for p in images:
            try:
                with Image.open(p) as im:
                    visual += 256 + ((im.width + 7) // 8) * ((im.height + 7) // 8)
            except OSError:
                visual += Path(p).stat().st_size
        return dict(text_bytes=text, images=len(images), reserved_tokens=text + visual + self.output)

    def fits(self, usage):
        return (
            usage["text_bytes"] <= self.text
            and usage["images"] <= self.images
            and usage["reserved_tokens"] <= self.tokens
        )
