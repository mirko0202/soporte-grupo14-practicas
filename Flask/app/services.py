import calendar
from itertools import count
from datetime import date

from .models import Habit


class HabitStore:
    def __init__(self):
        self._habits = []
        self._next_id = count(1)
        self._selected_day = date.today()

    def list_habits(self):
        return sorted(self._habits, key=lambda habit: habit.created_at, reverse=True)

    @property
    def selected_day(self):
        return self._selected_day

    def set_selected_day(self, selected_day):
        self._selected_day = selected_day

    def month_calendar(self):
        first_day = self._selected_day.replace(day=1)
        calendar_weeks = []

        for week in calendar.Calendar(firstweekday=0).monthdayscalendar(first_day.year, first_day.month):
            calendar_weeks.append([
                self._calendar_day_cell(first_day.replace(day=day_number)) if day_number else None
                for day_number in week
            ])

        return {
            "month_label": first_day.strftime("%B %Y"),
            "weekdays": ["Lu", "Ma", "Mi", "Ju", "Vi", "Sa", "Do"],
            "weeks": calendar_weeks,
        }

    def _calendar_day_cell(self, day):
        visible_habits = [habit for habit in self._habits if habit.visible_on(day)]
        completed_count = sum(1 for habit in visible_habits if habit.marked_on(day))
        return {
            "date": day,
            "habit_count": len(visible_habits),
            "completed_count": completed_count,
            "is_completed": bool(visible_habits) and completed_count == len(visible_habits),
        }

    def visible_habits(self, day=None):
        day = day or self._selected_day
        return [habit for habit in self.list_habits() if habit.visible_on(day)]

    def add_habit(self, title, description, target_days, created_on=None):
        title = title.strip()
        description = description.strip()

        if not title:
            raise ValueError("El nombre del habito es obligatorio.")

        if target_days <= 0:
            raise ValueError("Los dias objetivo deben ser mayores que cero.")

        habit = Habit(
            id=next(self._next_id),
            title=title,
            description=description,
            target_days=target_days,
            created_on=created_on or self._selected_day,
        )
        self._habits.append(habit)
        return habit

    def complete_habit(self, habit_id, completed_on=None):
        habit = self._find_habit(habit_id)
        if habit is None:
            raise LookupError("No se encontro el habito solicitado.")

        completed_on = completed_on or self._selected_day
        if not habit.visible_on(completed_on):
            raise ValueError("Este habito no esta activo en ese dia.")

        if habit.marked_on(completed_on):
            raise ValueError("Este habito ya fue marcado en ese dia.")

        if completed_on not in habit.completed_dates:
            habit.completed_dates.add(completed_on)
            habit.completed_days += 1
        return habit

    def delete_habit(self, habit_id):
        habit = self._find_habit(habit_id)
        if habit is None:
            raise LookupError("No se encontro el habito solicitado.")

        self._habits.remove(habit)

    def summary(self, day=None):
        visible_habits = self.visible_habits(day)
        total = len(visible_habits)
        completed = sum(1 for habit in visible_habits if habit.is_completed)
        active = total - completed
        return {
            "total": total,
            "completed": completed,
            "active": active,
            "selected_day": self._selected_day,
        }

    def _find_habit(self, habit_id):
        return next((habit for habit in self._habits if habit.id == habit_id), None)


store = HabitStore()
