from dataclasses import dataclass, field
from datetime import date, datetime, timedelta


@dataclass
class Habit:
    id: int
    title: str
    description: str
    target_days: int
    created_on: date = field(default_factory=date.today)
    completed_days: int = 0
    completed_dates: set[date] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def progress_percent(self) -> int:
        if self.target_days <= 0:
            return 0
        value = int((self.completed_days / self.target_days) * 100)
        return min(value, 100)

    @property
    def completion_ratio(self) -> int:
        return self.progress_percent

    @property
    def is_completed(self) -> bool:
        return self.completed_days >= self.target_days

    def current_streak(self, as_of: date) -> int:
        streak = 0
        current_day = as_of

        while current_day in self.completed_dates:
            streak += 1
            current_day = current_day.fromordinal(current_day.toordinal() - 1)

        return streak

    def best_streak(self) -> int:
        if not self.completed_dates:
            return 0

        best = 0
        streak = 0
        previous_day = None

        for current_day in sorted(self.completed_dates):
            if previous_day is not None and (current_day - previous_day).days == 1:
                streak += 1
            else:
                streak = 1

            best = max(best, streak)
            previous_day = current_day

        return best

    @property
    def expires_on(self) -> date:
        if self.target_days <= 1:
            return self.created_on
        return self.created_on + timedelta(days=self.target_days - 1)

    def visible_on(self, day: date) -> bool:
        return self.created_on <= day <= self.expires_on

    def marked_on(self, day: date) -> bool:
        return day in self.completed_dates
