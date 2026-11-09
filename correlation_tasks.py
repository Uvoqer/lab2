"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput

FEATURES = ["FSIQ", "VIQ", "PIQ", "Weight", "Height"]


def mri_correlation(df):
    corr = {}
    for feature in FEATURES:
        corr[feature] = float(df[feature].corr(df["MRI_Count"], method="pearson"))
    return corr


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    brainFrame = pd.read_csv(data.csv_path, delimiter="\t", na_values=["NA"])
    menDf = brainFrame[brainFrame["Gender"] == "Male"]
    womenDf = brainFrame[brainFrame["Gender"] == "Female"]

    men_corr = mri_correlation(menDf)
    women_corr = mri_correlation(womenDf)

    strongest = None
    best = -1
    for corr in (women_corr, men_corr):
        for feature, value in corr.items():
            if abs(value) > best:
                best = abs(value)
                strongest = feature

    return BrainCorrelationSummary(len(menDf), len(womenDf), women_corr, men_corr, strongest)


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

    res = analyze_brain_correlations(BrainDataInput("brainsize.txt"))
    test("Мужчин", res.men_count, 20)
    test("Женщин", res.women_count, 20)
    test("Признаки у мужчин", sorted(res.men_mri_correlation), sorted(FEATURES))
    test("Признаки у женщин", sorted(res.women_mri_correlation), sorted(FEATURES))
    test("Нет NaN в корреляциях", any(v != v for v in res.men_mri_correlation.values()), False)
    test("MRI_Count не участвует", res.strongest_mri_feature != "MRI_Count", True)
    print("Женщины:", res.women_mri_correlation)
    print("Мужчины:", res.men_mri_correlation)
    print("Самый сильный признак:", res.strongest_mri_feature)

    # маленький файл: у мужчин Height идеально коррелирует с MRI_Count
    small = (
        "Gender\tFSIQ\tVIQ\tPIQ\tWeight\tHeight\tMRI_Count\n"
        "Male\t100\t90\t110\t70\t170\t900\n"
        "Male\t120\t95\t100\tNA\t180\t1000\n"
        "Male\t110\t130\t105\t80\t190\t1100\n"
        "Female\t100\t100\t100\t50\t160\t800\n"
        "Female\t105\t101\t90\t55\t165\t850\n"
        "Female\t90\t99\t95\t60\tNA\t810\n"
    )
    path = os.path.join(tempfile.mkdtemp(), "small.txt")
    with open(path, "w") as f:
        f.write(small)
    res = analyze_brain_correlations(BrainDataInput(path))
    test("Мужчин в маленьком", res.men_count, 3)
    test("Женщин в маленьком", res.women_count, 3)
    test("Height у мужчин", round(res.men_mri_correlation["Height"], 6), 1.0)
    test("Weight с NA у мужчин", round(res.men_mri_correlation["Weight"], 6), 1.0)
    test("Самый сильный", res.strongest_mri_feature in ("Height", "Weight"), True)

    print("\nВсе тесты пройдены!")
