import hashlib
import hmac


def pseudonymize_player_reference(reference: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), reference.encode("utf-8"), hashlib.sha256).hexdigest()


def display_player_key(player_hash: str) -> str:
    return f"pp_{player_hash[:12]}"
