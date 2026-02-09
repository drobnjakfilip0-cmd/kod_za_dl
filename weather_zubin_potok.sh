# --- Notifikacija ---
MESSAGE="Zubin Potok ($TODAY)
Trenutno: $CURRENT_TEMP
Sutra: $FORECAST_TEMP"

osascript -e "display notification \"$MESSAGE\" with title \"Vremenska prognoza\""

