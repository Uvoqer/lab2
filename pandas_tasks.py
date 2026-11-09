"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    file = pd.read_csv(data.csv_path)

    row_count = len(file)

    missing = {}
    for col, count in file.isna().sum().items():
        missing[str(col)] = int(count)

    over_30 = int((file['Age'] > 30).sum())

    # mean() сам пропускает NaN, поэтому fillna(0) тут не нужен
    mean_age = {}
    for pclass, age in file.groupby('Pclass')['Age'].mean().items():
        mean_age[int(pclass)] = float(age)

    survival = {}
    for pclass, rate in file.groupby('Pclass')['Survived'].mean().items():
        survival[int(pclass)] = float(rate)

    fares = file['Fare'].dropna().sort_values(ascending=False).head(5)
    highest_fares = [float(f) for f in fares]

    return TitanicSummary(row_count, missing, over_30, mean_age, survival, highest_fares)


# тесты

def test(name, result, answer):
    print(name, "=", result)
    if result == answer:
        print("Тест успешно пройден!")
    else:
        print("Тест не пройден, ожидалось:", answer)
        raise AssertionError


if __name__ == "__main__":
    import os
    import tempfile

    res = analyze_titanic(TitanicInput("titanic.csv"))
    test("Число строк", res.row_count, 891)
    test("Пропуски Age", res.missing_by_column['Age'], 177)
    test("Пропуски Cabin", res.missing_by_column['Cabin'], 687)
    test("Пропуски Embarked", res.missing_by_column['Embarked'], 2)
    test("Пропуски Name", res.missing_by_column['Name'], 0)
    test("Ключи классов", sorted(res.mean_age_by_pclass), [1, 2, 3])
    test("Топ-5 Fare по убыванию", res.highest_fares, sorted(res.highest_fares, reverse=True))
    test("Количество тарифов", len(res.highest_fares), 5)
    print("Старше 30:", res.adults_over_30_count)
    print("Средний возраст:", res.mean_age_by_pclass)
    print("Доля выживших:", res.survival_rate_by_pclass)
    print("Топ-5 Fare:", res.highest_fares)

    # маленький файл, где всё можно посчитать руками
    small = (
        "PassengerId,Survived,Pclass,Name,Age,Fare\n"
        "1,1,1,a,40,100\n"
        "2,0,1,b,,50\n"
        "3,0,3,c,20,7\n"
        "4,1,3,d,31,8\n"
        "5,0,3,e,30,9\n"
        "6,1,2,f,60,10\n"
    )
    path = os.path.join(tempfile.mkdtemp(), "small.csv")
    with open(path, "w") as f:
        f.write(small)
    res = analyze_titanic(TitanicInput(path))
    test("Строк в маленьком", res.row_count, 6)
    test("Пропуски Age", res.missing_by_column['Age'], 1)
    test("Старше 30 (30 не считается)", res.adults_over_30_count, 3)
    test("Средний возраст без нулей", res.mean_age_by_pclass, {1: 40.0, 2: 60.0, 3: 27.0})
    test("Доля выживших", res.survival_rate_by_pclass, {1: 0.5, 2: 1.0, 3: 1 / 3})
    test("Топ-5 Fare", res.highest_fares, [100.0, 50.0, 10.0, 9.0, 8.0])

    print("\nВсе тесты пройдены!")
