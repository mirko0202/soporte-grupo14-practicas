from datetime import date

from flask import Blueprint, flash, redirect, render_template, request, url_for

from .services import store

tracker_bp = Blueprint("tracker", __name__)


@tracker_bp.route("/", methods=["GET", "POST"])
def index():
    selected_day_raw = request.args.get("day")
    if selected_day_raw:
        try:
            store.set_selected_day(date.fromisoformat(selected_day_raw))
        except ValueError:
            flash("La fecha seleccionada no es valida.", "danger")

    if request.method == "POST":
        try:
            title = request.form.get("title", "")
            description = request.form.get("description", "")
            target_days = int(request.form.get("target_days", 0))
            store.add_habit(title, description, target_days, created_on=store.selected_day)
            flash("Habito creado correctamente.", "success")
        except (ValueError, TypeError) as error:
            flash(str(error), "danger")
        return redirect(url_for("tracker.index", day=store.selected_day.isoformat()))

    habits = store.visible_habits()
    summary = store.summary()
    calendar = store.month_calendar()
    return render_template("index.html", habits=habits, summary=summary, calendar=calendar)


@tracker_bp.post("/habits/<int:habit_id>/complete")
def complete_habit(habit_id):
    try:
        completed_on = date.fromisoformat(request.form.get("completed_on", store.selected_day.isoformat()))
        store.complete_habit(habit_id, completed_on=completed_on)
        flash("Progreso actualizado.", "success")
    except LookupError as error:
        flash(str(error), "danger")
    except ValueError:
        flash("No se puede marcar ese dia. Revisa si ya estaba marcado o si el habito esta activo.", "warning")
    return redirect(url_for("tracker.index", day=store.selected_day.isoformat()))


@tracker_bp.post("/habits/<int:habit_id>/delete")
def delete_habit(habit_id):
    try:
        store.delete_habit(habit_id)
        flash("Habito eliminado.", "warning")
    except LookupError as error:
        flash(str(error), "danger")
    return redirect(url_for("tracker.index", day=store.selected_day.isoformat()))


@tracker_bp.post("/calendar/day")
def select_day():
    selected_day_raw = request.form.get("selected_day", "")
    try:
        store.set_selected_day(date.fromisoformat(selected_day_raw))
        flash("Dia simulado actualizado.", "success")
    except ValueError:
        flash("La fecha seleccionada no es valida.", "danger")
    return redirect(url_for("tracker.index", day=store.selected_day.isoformat()))
