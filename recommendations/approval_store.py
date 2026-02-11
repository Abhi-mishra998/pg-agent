import json
from pathlib import Path
from typing import Dict, Any

STORE_DIR = Path("data/approvals")
STORE_DIR.mkdir(parents=True, exist_ok=True)


def _card_path(card_id: str) -> Path:
    return STORE_DIR / f"{card_id}.json"


def save_card(card_dict: Dict[str, Any]) -> None:
    """Persist a full review card (atomic overwrite)."""
    p = _card_path(card_dict.get("card_id", "card_unknown"))
    p.write_text(json.dumps(card_dict, indent=2))


def load_card(card_id: str) -> Dict[str, Any]:
    p = _card_path(card_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def append_audit_entry(card_id: str, entry: Dict[str, Any]) -> None:
    """Append an audit entry to the stored card. Creates file if missing."""
    card = load_card(card_id)
    if not card:
        # create a minimal card wrapper
        card = {"card_id": card_id, "audit_trail": {"trail_id": f"trail_{card_id}", "review_card_id": card_id, "entries": []}}

    audit = card.get("audit_trail") or {"entries": []}
    entries = audit.get("entries", [])
    entries.append(entry)
    audit["entries"] = entries
    card["audit_trail"] = audit
    save_card(card)
