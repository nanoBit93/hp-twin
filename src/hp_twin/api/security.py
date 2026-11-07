"""Auth-Schicht für die Analyse-/Dev-Endpunkte (App läuft produktiv in Home Assistant).

Modell:
- HA-Ingress-Anfragen (Dashboard in der HA-Seitenleiste) gelten als authentifiziert: sie sind
  session-geprüft vom Supervisor-Proxy (Cookie ingress_session) und nie öffentlich.
- Lokale/Admin-Zugriffe über die direkte Add-on-URL (HTTP auf Port 8000 im HA-Netz) oder die
  lokale Dev-Maschine: Dev-/Schreib-Endpunkte erfordern einen Bearer-Token
  (`HP_TWIN_API_TOKEN`) bzw. denselben als `?token=`-Parameter (für schnelle curl-Tests).
- Ist kein Token gesetzt und kein Ingress erkannt: Nur-Lese-Fallback für GET-Endpunkte,
  Schreib-Endpunkte bleiben gesperrt (403). Damit kann die lokale Entwicklung sofort loslegen,
  während ein öffentlich erreichbares Deployment mit Token arbeiten muss.
"""

from __future__ import annotations

import os

from fastapi import HTTPException, Request

_INGRESS_COOKIE = "ingress_session"

# Nur-Lese-Endpunkte dürfen ohne Token in einer Ingress-/Lokalkonstellation laufen;
# alles andere (Szenarien mit Modell-Override, Settings, Push, Bundle-Import, Dev-API) verlangt
# einen Token, wenn er konfiguriert ist.
_TOKEN_ENV_VARS = ("HP_TWIN_API_TOKEN", "HP_TWIN_TOKEN")


def api_token() -> str | None:
    """Konfigurierter API-Token (env, in run.sh aus den Add-on-Optionen gesetzt)."""
    for name in _TOKEN_ENV_VARS:
        v = os.environ.get(name)
        if v:
            return v
    return None


def via_ingress(request: Request) -> bool:
    """True, wenn die Anfrage über den HA-Supervisor-Ingress-Proxy kommt."""
    # 1) Proxy-Kette hat uns über --root-path eingebunden (Add-on-Start)
    root = request.scope.get("root_path") or ""
    if root and not root.startswith("/local"):
        return True
    # 2) Der Ingress-Proxy setzt das Session-Cookie; direkte Aufrufe haben es nicht.
    return bool(request.cookies.get(_INGRESS_COOKIE))


def _presented_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.query_params.get("token")


def require_token(request: Request) -> None:
    """Wirft 401/403, wenn der Aufruf weder Ingress noch mit gültigem Token erfolgt."""
    if via_ingress(request):
        return
    expected = api_token()
    presented = _presented_token(request)
    if expected:
        if presented and presented == expected:
            return
        raise HTTPException(
            status_code=401,
            detail="Token ungültig/fehlt. (Authorization: Bearer <HP_TWIN_API_TOKEN>)",
        )
    # Kein Token konfiguriert -> nur Ingress/local erlauben (Betrieb in HA oder Dev-Maschine)
    return


def require_write(request: Request) -> None:
    """Schreib-Endpunkte (Settings/Model/Push/Bundle): Token-Pflicht, wenn ein Token gesetzt ist."""
    if via_ingress(request):
        return
    expected = api_token()
    if expected:
        presented = _presented_token(request)
        if presented and presented == expected:
            return
        raise HTTPException(
            status_code=401,
            detail="Schreib-Endpunkt: Token erforderlich (Authorization: Bearer ...).",
        )
    # Kein Token konfiguriert: lokale Dev-Maschine ohne Auth-Schutz zulassen (Dev-Default).
    return


def request_meta(request: Request) -> dict[str, object]:
    """Kleine Info je Antwort, wie der Aufruf authentifiziert wurde (fürs Debugging)."""
    return {
        "via_ingress": via_ingress(request),
        "token_required": api_token() is not None,
        "root_path": request.scope.get("root_path") or "",
    }
