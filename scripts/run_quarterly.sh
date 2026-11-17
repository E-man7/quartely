#!/bin/bash
# Keep the Quarterly bot running: prevents idle sleep (caffeinate) and restarts it after a crash.
#
#   scripts/run_quarterly.sh              # paper (LIVE_TRADING is never passed through)
#   scripts/run_quarterly.sh --dry-run
#   kill $(cat logs/quarterly_runner.pid) # stop: the bot shuts down cleanly and the loop ends
#
# A clean stop (exit code 0, e.g. SIGTERM/Ctrl+C) ends the loop; any other exit restarts after 30 s.
# caffeinate -i stops idle sleep only: closing the lid or a reboot still stops the bot.
cd "$(dirname "$0")/.." || exit 1
mkdir -p logs
# Only one runner (and so one bot) at a time: the login item and a manual start must not both trade
if [ -f logs/quarterly_runner.pid ] && kill -0 "$(cat logs/quarterly_runner.pid)" 2>/dev/null \
   && ps -p "$(cat logs/quarterly_runner.pid)" -o command= | grep -q run_quarterly; then
  echo "$(date '+%F %T') runner: already running (pid $(cat logs/quarterly_runner.pid)), not starting another"
  exit 0
fi
echo $$ > logs/quarterly_runner.pid

child=""
stop() {
  echo "$(date '+%F %T') runner: stop requested"
  [ -n "$child" ] && kill -TERM "$child" 2>/dev/null && wait "$child"
  rm -f logs/quarterly_runner.pid
  exit 0
}
trap stop TERM INT

while true; do
  echo "$(date '+%F %T') runner: starting bot $*"
  env -u LIVE_TRADING .venv/bin/python -m quarterly live "$@" &
  child=$!
  echo "$child" > logs/quarterly_live.pid
  caffeinate -i -w "$child" &          # no idle sleep while the bot runs
  wait "$child"
  code=$?
  child=""
  if [ "$code" -eq 0 ]; then
    echo "$(date '+%F %T') runner: bot exited cleanly"
    rm -f logs/quarterly_runner.pid
    exit 0
  fi
  echo "$(date '+%F %T') runner: bot exited with code $code, restarting in 30 s"
  sleep 30
done
