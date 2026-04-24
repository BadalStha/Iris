"""
skill_weather.py — Weather and forecast
Priority 30.
"""

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

METADATA = {
    "name":        "Weather",
    "version":     "1.0",
    "description": "Current weather and forecast via open-meteo",
    "author":      "iris",
}

WEATHER_PHRASES = ["weather", "forecast", "temperature", "rain", "snow", "conditions"]

_WMO = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "freezing fog", 51: "light drizzle", 53: "moderate drizzle",
    55: "dense drizzle", 61: "light rain", 63: "moderate rain", 65: "heavy rain",
    71: "light snow", 73: "moderate snow", 75: "heavy snow",
    80: "rain showers", 81: "strong showers", 82: "violent showers",
    95: "thunderstorm", 96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


def _fetch(url):
    req = Request(url, headers={"User-Agent": "Iris/1.0"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode())


def _norm_loc(text):
    cleaned = re.sub(
        r"\b(weather|forecast|temperature|today|tomorrow|now|current|please|check|tell me|what's|whats|show|give me)\b",
        " ", text.lower(),
    )
    cleaned = re.sub(r"[^a-z0-9,\s.-]", " ", cleaned)
    return " ".join(cleaned.split()).strip(" ,.-")


def _extract_location(command):
    for pat in [
        r"\bweather\s+(?:in|for|at)\s+(.+)$",
        r"\bforecast\s+(?:in|for|at)\s+(.+)$",
        r"\btemperature\s+(?:in|for|at)\s+(.+)$",
    ]:
        m = re.search(pat, command, re.IGNORECASE)
        if m:
            loc = _norm_loc(m.group(1))
            if loc:
                return loc
    return None


def _candidates(query):
    base = _norm_loc(query)
    seen, out = set(), []
    def add(v):
        v = _norm_loc(v)
        if v and v not in seen:
            seen.add(v); out.append(v)
    add(base)
    if "," in base:
        parts = [p.strip() for p in base.split(",") if p.strip()]
        if len(parts) > 1:
            add(", ".join(parts[-2:])); add(parts[-1]); add(parts[0])
    for word in ["district", "province", "zone", "state"]:
        if word in base:
            add(base.replace(word, ""))
    return out


def _get_weather(location_query, with_forecast=True):
    if not location_query:
        return None, "I need a location."
    try:
        results = []
        for cand in _candidates(location_query):
            geo = _fetch(
                f"https://geocoding-api.open-meteo.com/v1/search?"
                f"name={quote_plus(cand)}&count=1&language=en&format=json"
            )
            results = geo.get("results") or []
            if results:
                break
        if not results:
            return None, f"I couldn't find {location_query}."

        p = results[0]
        lat, lon = p["latitude"], p["longitude"]
        display = ", ".join(x for x in [p.get("name"), p.get("admin1"), p.get("country")] if x)

        data = _fetch(
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m,relative_humidity_2m"
            "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            "&timezone=auto"
        )
        cur = data.get("current") or {}
        temp = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        wind = cur.get("wind_speed_10m")
        hum = cur.get("relative_humidity_2m")
        desc = _WMO.get(cur.get("weather_code", -1), "unknown")

        line = f"Current weather in {display}: {temp:.0f}°C" if temp is not None else f"In {display}: {desc}"
        if temp is not None:
            if feels is not None: line += f", feels like {feels:.0f}°C"
            line += f", {desc}"
            if wind is not None:  line += f", wind {wind:.0f} km/h"
            if hum is not None:   line += f", humidity {hum:.0f}%"
        parts = [line + "."]

        if with_forecast:
            daily = data.get("daily") or {}
            dates = daily.get("time") or []
            highs = daily.get("temperature_2m_max") or []
            lows  = daily.get("temperature_2m_min") or []
            rain  = daily.get("precipitation_probability_max") or []
            codes = daily.get("weather_code") or []
            bits = []
            for i in range(min(2, len(dates))):
                label = "Today" if i == 0 else "Tomorrow"
                d = f"{label}: {_WMO.get(codes[i], 'weather')}"
                if i < len(highs) and i < len(lows):
                    d += f", high {highs[i]:.0f}°C low {lows[i]:.0f}°C"
                if i < len(rain):
                    d += f", rain {rain[i]:.0f}%"
                bits.append(d)
            if bits:
                parts.append("Forecast: " + "; ".join(bits) + ".")

        return " ".join(parts), None
    except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError) as exc:
        return None, f"Couldn't fetch weather: {str(exc)[:80]}"


def _match_weather(norm):
    return any(p in norm for p in WEATHER_PHRASES)


def _handle_weather(command, ctx):
    location = _extract_location(command)
    if not location:
        location = ctx["memory"].get("user", {}).get("location")
    if not location:
        return "Which location should I check?"
    include_fc = not any(p in command for p in ["current weather", "right now only", "now only"])
    report, err = _get_weather(location, with_forecast=include_fc)
    return report or err


INTENTS = [
    {"name": "weather", "priority": 30, "match": _match_weather, "handle": _handle_weather},
]
