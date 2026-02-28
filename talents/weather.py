import requests
from talents.base import BaseTalent


# WMO weather interpretation codes -> human descriptions
_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Foggy", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherTalent(BaseTalent):
    name = "weather"
    description = "Get current weather conditions from configurable providers"
    keywords = ["weather", "temperature", "forecast", "rain", "snow",
                "humid", "wind", "degrees", "cold", "hot", "warm"]
    examples = [
        "what's the weather like in New York",
        "is it going to rain today",
        "how cold is it outside",
        "what's the temperature right now",
    ]
    priority = 75  # above web_search (60), below news (80)

    _WEATHER_PHRASES = [
        "weather", "temperature", "forecast", "how hot", "how cold",
        "how warm", "raining", "snowing", "rain", "snow", "wind",
        "humid", "degrees", "outside", "feels like",
    ]

    _SYSTEM_PROMPT = (
        "You are a weather reporter. "
        "Using ONLY the weather data provided, give a natural conversational summary. "
        "Include the current temperature, what it feels like, conditions, today's high "
        "and low, humidity, wind, and any notable precipitation. "
        "Be concise but complete (3-5 sentences). Do NOT add information not in the data."
    )

    _SCOPE_DAYS = {"today": 1, "tomorrow": 2, "week": 7, "weekend": 7}

    def get_config_schema(self) -> dict:
        return {
            "fields": [
                {"key": "provider", "label": "Weather Provider",
                 "type": "choice", "default": "Open-Meteo",
                 "choices": ["Open-Meteo", "OpenWeatherMap", "WeatherAPI"]},
                {"key": "api_key", "label": "API Key",
                 "type": "password", "default": ""},
                {"key": "default_location", "label": "Default Location",
                 "type": "string", "default": ""},
                {"key": "temperature_unit", "label": "Temperature Unit",
                 "type": "choice", "default": "fahrenheit",
                 "choices": ["fahrenheit", "celsius"]},
            ]
        }

    def can_handle(self, command: str) -> bool:
        return self.keyword_match(command)

    def execute(self, command: str, context: dict) -> dict:
        llm = context["llm"]
        location = (
            self._extract_arg(llm, command, "location")
            or self._config.get("default_location", "")
        )
        print(f"   [Weather] Extracted location: {repr(location)}")

        time_scope = (
            self._extract_arg(
                llm, command, "time period",
                options=["today", "tomorrow", "week", "weekend"],
                fallback="today",
            ) or "today"
        ).lower()
        forecast_days = self._SCOPE_DAYS.get(time_scope, 1)
        print(f"   [Weather] Time scope: {time_scope!r}  forecast_days={forecast_days}")

        provider = self._config.get("provider", "Open-Meteo")

        # Geocode the location to lat/lon (needed for Open-Meteo and OWM)
        geo = self._geocode(location)
        if geo is None:
            return {
                "success": False,
                "response": f"Sorry, I couldn't find a location matching '{location or 'your area'}'.",
                "actions_taken": [{"action": "weather_lookup", "location": location}],
                "spoken": False
            }

        # Fetch weather data from selected provider
        weather_data = self._fetch_weather(geo, provider, forecast_days)
        if weather_data is None:
            return {
                "success": False,
                "response": f"Sorry, I couldn't get weather data for {geo['name']} via {provider}.",
                "actions_taken": [{"action": "weather_lookup", "location": geo["name"],
                                   "provider": provider}],
                "spoken": False
            }

        # Format the raw data into a readable block for the LLM
        formatted = self._format_weather(weather_data, geo, provider, forecast_days)

        print(f"   -> Weather data for '{geo['name']}' via {provider}:")
        print(f"   -> {formatted[:400]}...")
        user_message = (
            f"=== CURRENT WEATHER DATA ===\n"
            f"{formatted}\n"
            f"=== END WEATHER DATA ===\n\n"
            f"User asked: {command}\n"
            f"Time focus: {time_scope}\n\n"
            f"Summarize the weather using ONLY the data above. "
            f"Focus on the {time_scope} data."
        )

        response = llm.generate(
            user_message,
            system_prompt=self._SYSTEM_PROMPT,
            temperature=0.3,
        )

        return {
            "success": True,
            "response": response,
            "actions_taken": [{"action": "weather_lookup", "location": geo["name"],
                               "provider": provider}],
            "spoken": False
        }

    # ── Provider dispatch ──────────────────────────────────────────

    def _fetch_weather(self, geo, provider, forecast_days=1):
        """Fetch weather data from the selected provider."""
        providers = {
            "Open-Meteo": self._fetch_open_meteo,
            "OpenWeatherMap": self._fetch_openweathermap,
            "WeatherAPI": self._fetch_weatherapi,
        }
        fetch_fn = providers.get(provider, self._fetch_open_meteo)
        # Only Open-Meteo supports multi-day natively here; OWM/WeatherAPI
        # use their existing single-day endpoints and ignore forecast_days.
        if provider == "Open-Meteo":
            return fetch_fn(geo, forecast_days)
        return fetch_fn(geo)

    def _format_weather(self, data, geo, provider, forecast_days=1):
        """Format weather data into a readable block for the LLM."""
        formatters = {
            "Open-Meteo": self._format_open_meteo,
            "OpenWeatherMap": self._format_owm,
            "WeatherAPI": self._format_weatherapi,
        }
        fmt_fn = formatters.get(provider, self._format_open_meteo)
        if provider == "Open-Meteo":
            return fmt_fn(data, geo, forecast_days)
        return fmt_fn(data, geo)

    # ── Open-Meteo (default, no key) ──────────────────────────────

    def _fetch_open_meteo(self, geo, forecast_days=1):
        """Fetch current weather + forecast from Open-Meteo (free, no API key)."""
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": geo["lat"],
                "longitude": geo["lon"],
                "current": ",".join([
                    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                    "weather_code", "wind_speed_10m", "wind_direction_10m",
                    "wind_gusts_10m", "precipitation", "cloud_cover",
                    "surface_pressure", "is_day",
                ]),
                "daily": ",".join([
                    "weather_code",
                    "temperature_2m_max", "temperature_2m_min",
                    "apparent_temperature_max", "apparent_temperature_min",
                    "sunrise", "sunset", "precipitation_sum",
                    "wind_speed_10m_max", "uv_index_max",
                ]),
                "temperature_unit": self._config.get("temperature_unit", "fahrenheit"),
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "auto",
                "forecast_days": forecast_days,
            }
            print(f"   -> Fetching Open-Meteo: lat={geo['lat']}, lon={geo['lon']}")
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"   -> Open-Meteo returned status {resp.status_code}")
                return None
        except Exception as e:
            print(f"   -> ERROR fetching Open-Meteo: {e}")
            return None

    def _format_open_meteo(self, data, geo, forecast_days=1):
        """Format Open-Meteo JSON into a readable block for the LLM."""
        try:
            current = data.get("current", {})
            daily = data.get("daily", {})

            temp_f = current.get("temperature_2m", "?")
            feels_f = current.get("apparent_temperature", "?")
            humidity = current.get("relative_humidity_2m", "?")
            wind_mph = current.get("wind_speed_10m", "?")
            wind_dir = self._wind_direction(current.get("wind_direction_10m", 0))
            wind_gust = current.get("wind_gusts_10m", "?")
            precip = current.get("precipitation", 0)
            cloud_cover = current.get("cloud_cover", "?")
            pressure = current.get("surface_pressure", "?")
            is_day = current.get("is_day", 1)
            weather_code = current.get("weather_code", 0)
            conditions = _WMO_CODES.get(weather_code, f"Code {weather_code}")
            day_night = "Daytime" if is_day else "Nighttime"

            unit = "°F" if self._config.get("temperature_unit", "fahrenheit") == "fahrenheit" else "°C"
            formatted = (
                f"Location: {geo['name']}\n"
                f"Time of day: {day_night}\n"
                f"Conditions: {conditions}\n"
                f"Temperature: {temp_f}{unit}\n"
                f"Feels like: {feels_f}{unit}\n"
                f"Humidity: {humidity}%\n"
                f"Wind: {wind_mph} mph from {wind_dir} (gusts {wind_gust} mph)\n"
                f"Cloud cover: {cloud_cover}%\n"
                f"Precipitation: {precip} inches\n"
                f"Pressure: {pressure} hPa\n"
            )

            if daily:
                max_temps  = daily.get("temperature_2m_max", [])
                min_temps  = daily.get("temperature_2m_min", [])
                sunrises   = daily.get("sunrise", [])
                sunsets    = daily.get("sunset", [])
                uv_maxes   = daily.get("uv_index_max", [])
                precip_sums = daily.get("precipitation_sum", [])
                dates      = daily.get("time", [])
                codes      = daily.get("weather_code", [])
                wind_maxes = daily.get("wind_speed_10m_max", [])

                if forecast_days == 1:
                    # Single-day: today's summary (original behaviour)
                    if max_temps:
                        formatted += f"\nToday's high: {max_temps[0]}{unit}\n"
                    if min_temps:
                        formatted += f"Today's low: {min_temps[0]}{unit}\n"
                    if sunrises:
                        sunrise_str = sunrises[0].split("T")[1] if "T" in sunrises[0] else sunrises[0]
                        formatted += f"Sunrise: {sunrise_str}\n"
                    if sunsets:
                        sunset_str = sunsets[0].split("T")[1] if "T" in sunsets[0] else sunsets[0]
                        formatted += f"Sunset: {sunset_str}\n"
                    if uv_maxes:
                        formatted += f"UV Index (max): {uv_maxes[0]}\n"
                    if precip_sums:
                        formatted += f"Total precipitation today: {precip_sums[0]} inches\n"
                else:
                    # Multi-day: one row per day
                    formatted += "\nForecast:\n"
                    days_available = min(forecast_days, len(dates) if dates else forecast_days)
                    for i in range(days_available):
                        label = dates[i] if dates else f"Day {i + 1}"
                        cond  = _WMO_CODES.get(int(codes[i]) if codes else 0, "Unknown")
                        high  = max_temps[i]   if max_temps   else "?"
                        low   = min_temps[i]   if min_temps   else "?"
                        rain  = precip_sums[i] if precip_sums else 0
                        wind  = wind_maxes[i]  if wind_maxes  else "?"
                        formatted += (
                            f"  {label}: High {high}{unit} / Low {low}{unit}, "
                            f"{cond}, Wind max {wind} mph, Rain {rain} in\n"
                        )

            return formatted

        except (KeyError, IndexError) as e:
            print(f"   -> Error parsing Open-Meteo data: {e}")
            return str(data)

    # ── OpenWeatherMap ─────────────────────────────────────────────

    def _fetch_openweathermap(self, geo):
        """Fetch current weather from OpenWeatherMap (requires API key).

        Free tier: 60 calls/minute, current weather only.
        Get a key at https://openweathermap.org/api
        """
        api_key = self._config.get("api_key", "")
        if not api_key:
            print("   -> OpenWeatherMap requires an API key")
            return None

        try:
            unit_param = "imperial" if self._config.get("temperature_unit", "fahrenheit") == "fahrenheit" else "metric"
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {
                "lat": geo["lat"],
                "lon": geo["lon"],
                "appid": api_key,
                "units": unit_param,
            }
            print(f"   -> Fetching OpenWeatherMap: lat={geo['lat']}, lon={geo['lon']}")
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 401:
                print("   -> OpenWeatherMap API key invalid")
                return None
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"   -> OpenWeatherMap returned status {resp.status_code}")
                return None
        except Exception as e:
            print(f"   -> ERROR fetching OpenWeatherMap: {e}")
            return None

    def _format_owm(self, data, geo):
        """Format OpenWeatherMap JSON into a readable block."""
        try:
            main = data.get("main", {})
            wind_data = data.get("wind", {})
            weather_list = data.get("weather", [{}])
            clouds = data.get("clouds", {})
            sys_data = data.get("sys", {})

            conditions = weather_list[0].get("description", "Unknown").title() if weather_list else "Unknown"
            temp = main.get("temp", "?")
            feels = main.get("feels_like", "?")
            humidity = main.get("humidity", "?")
            pressure = main.get("pressure", "?")
            wind_speed = wind_data.get("speed", "?")
            wind_deg = wind_data.get("deg", 0)
            wind_gust = wind_data.get("gust", "?")
            cloud_cover = clouds.get("all", "?")

            unit = "°F" if self._config.get("temperature_unit", "fahrenheit") == "fahrenheit" else "°C"
            formatted = (
                f"Location: {geo['name']}\n"
                f"Conditions: {conditions}\n"
                f"Temperature: {temp}{unit}\n"
                f"Feels like: {feels}{unit}\n"
                f"Humidity: {humidity}%\n"
                f"Wind: {wind_speed} mph from {self._wind_direction(wind_deg)}"
                f" (gusts {wind_gust} mph)\n"
                f"Cloud cover: {cloud_cover}%\n"
                f"Pressure: {pressure} hPa\n"
            )

            # Min/max from OWM main block
            temp_min = main.get("temp_min")
            temp_max = main.get("temp_max")
            if temp_min is not None:
                formatted += f"Today's low: {temp_min}{unit}\n"
            if temp_max is not None:
                formatted += f"Today's high: {temp_max}{unit}\n"

            return formatted

        except (KeyError, IndexError) as e:
            print(f"   -> Error parsing OWM data: {e}")
            return str(data)

    # ── WeatherAPI.com ─────────────────────────────────────────────

    def _fetch_weatherapi(self, geo):
        """Fetch current weather from WeatherAPI.com (requires API key).

        Free tier: 1M calls/month, current + forecast.
        Get a key at https://www.weatherapi.com/
        """
        api_key = self._config.get("api_key", "")
        if not api_key:
            print("   -> WeatherAPI requires an API key")
            return None

        try:
            url = "https://api.weatherapi.com/v1/current.json"
            params = {
                "key": api_key,
                "q": f"{geo['lat']},{geo['lon']}",
                "aqi": "no",
            }
            print(f"   -> Fetching WeatherAPI: lat={geo['lat']}, lon={geo['lon']}")
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code == 403:
                print("   -> WeatherAPI key invalid or expired")
                return None
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"   -> WeatherAPI returned status {resp.status_code}")
                return None
        except Exception as e:
            print(f"   -> ERROR fetching WeatherAPI: {e}")
            return None

    def _format_weatherapi(self, data, geo):
        """Format WeatherAPI.com JSON into a readable block."""
        try:
            current = data.get("current", {})
            condition = current.get("condition", {}).get("text", "Unknown")

            use_f = self._config.get("temperature_unit", "fahrenheit") == "fahrenheit"
            unit = "°F" if use_f else "°C"
            temp = current.get("temp_f" if use_f else "temp_c", "?")
            feels = current.get("feelslike_f" if use_f else "feelslike_c", "?")
            humidity = current.get("humidity", "?")
            wind_mph = current.get("wind_mph", "?")
            wind_dir = current.get("wind_dir", "?")
            wind_gust = current.get("gust_mph", "?")
            cloud_cover = current.get("cloud", "?")
            pressure = current.get("pressure_mb", "?")
            precip = current.get("precip_in", 0)
            uv = current.get("uv", "?")
            is_day = current.get("is_day", 1)
            day_night = "Daytime" if is_day else "Nighttime"

            formatted = (
                f"Location: {geo['name']}\n"
                f"Time of day: {day_night}\n"
                f"Conditions: {condition}\n"
                f"Temperature: {temp}{unit}\n"
                f"Feels like: {feels}{unit}\n"
                f"Humidity: {humidity}%\n"
                f"Wind: {wind_mph} mph from {wind_dir} (gusts {wind_gust} mph)\n"
                f"Cloud cover: {cloud_cover}%\n"
                f"Precipitation: {precip} inches\n"
                f"Pressure: {pressure} hPa\n"
                f"UV Index: {uv}\n"
            )

            return formatted

        except (KeyError, IndexError) as e:
            print(f"   -> Error parsing WeatherAPI data: {e}")
            return str(data)

    # ── Geocoding (shared across providers) ────────────────────────

    def _extract_location(self, command):
        """Pull a location out of the command, default to empty (auto-detect)."""
        cmd = command.lower()
        for phrase in [
            "what's the weather like in", "what is the weather like in",
            "what's the weather in", "what is the weather in",
            "how's the weather in", "how is the weather in",
            "what's the temperature in", "what is the temperature in",
            "how hot is it in", "how cold is it in", "how warm is it in",
            "weather in", "weather for", "weather at",
            "temperature in", "temperature for", "temperature at",
            "forecast for", "forecast in",
        ]:
            if phrase in cmd:
                location = cmd.split(phrase, 1)[1].strip()
                location = location.rstrip("?.!")
                if location:
                    return location

        for prep in [" in ", " for ", " at ", " near "]:
            if prep in cmd:
                location = cmd.split(prep)[-1].strip().rstrip("?.!")
                if location and location not in [
                    "today", "tonight", "now", "right now",
                    "this week", "tomorrow", "the morning",
                ]:
                    return location

        return ""

    def _geocode(self, location):
        """Use Open-Meteo geocoding API to resolve a location name to lat/lon."""
        try:
            if not location:
                return self._geolocate_by_ip()

            queries = [location]
            no_commas = location.replace(",", " ").strip()
            if no_commas != location:
                queries.append(no_commas)
            parts = [p.strip() for p in location.replace(",", " ").split() if p.strip()]
            if len(parts) > 1:
                queries.append(parts[0])

            url = "https://geocoding-api.open-meteo.com/v1/search"

            for query in queries:
                params = {"name": query, "count": 5, "language": "en", "format": "json"}
                print(f"   -> Geocoding: '{query}'")
                resp = requests.get(url, params=params, timeout=10)

                if resp.status_code != 200:
                    continue

                data = resp.json()
                results = data.get("results")
                if not results:
                    continue

                best = self._best_geocode_match(results, location)

                name = best.get("name", location)
                admin = best.get("admin1", "")
                country = best.get("country", "")
                full_name = name
                if admin:
                    full_name += f", {admin}"
                if country:
                    full_name += f", {country}"

                result = {
                    "name": full_name,
                    "lat": best["latitude"],
                    "lon": best["longitude"],
                    "timezone": best.get("timezone", "auto"),
                }
                print(f"   -> Resolved: {full_name} ({result['lat']}, {result['lon']})")
                return result

            print(f"   -> All geocoding attempts failed for '{location}'")
            return None

        except Exception as e:
            print(f"   -> ERROR in geocoding: {e}")
            return None

    @staticmethod
    def _best_geocode_match(results, original_query):
        """Pick the geocode result that best matches the user's original query."""
        query_lower = original_query.lower()
        for r in results:
            admin = (r.get("admin1") or "").lower()
            country = (r.get("country") or "").lower()
            if admin and admin in query_lower:
                return r
            if country and country in query_lower:
                return r
        return results[0]

    def _geolocate_by_ip(self):
        """Fallback: get approximate location from IP address."""
        try:
            print("   -> No location given, detecting by IP...")
            resp = requests.get("https://ipapi.co/json/", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                name = f"{data.get('city', '?')}, {data.get('region', '?')}, {data.get('country_name', '?')}"
                return {
                    "name": name,
                    "lat": data["latitude"],
                    "lon": data["longitude"],
                    "timezone": data.get("timezone", "auto"),
                }
        except Exception as e:
            print(f"   -> IP geolocation failed: {e}")
        return None

    @staticmethod
    def _wind_direction(degrees):
        """Convert wind degrees to compass direction."""
        dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        try:
            idx = round(float(degrees) / 22.5) % 16
            return dirs[idx]
        except (ValueError, TypeError):
            return "?"
