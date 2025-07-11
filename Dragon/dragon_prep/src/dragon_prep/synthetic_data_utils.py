#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import numpy as np


def add_noise(report: str) -> str:
    """
    Introduce noise to the report by randomly removing and swapping spaces/letters.
    """
    # randomly swap some letters with its neighbouring letter (p = 0.01)
    # (this is a very simple way to introduce typos)
    # note: the first and last letter of the report are not swapped (easier to implement)
    report_pieces = list(report)
    for i in range(len(report_pieces)):
        if np.random.rand() < 0.01:
            neighbour = np.random.choice([-1, 1])
            if i + neighbour >= 0 and i + neighbour < len(report_pieces):
                report_pieces[i], report_pieces[i + neighbour] = report_pieces[i + neighbour], report_pieces[i]

    report = "".join(report_pieces)

    # randomly remove some words (p = 0.025)
    report_pieces = report.split(" ")
    report_pieces = [piece for piece in report_pieces if np.random.rand() > 0.025]

    # reconstruct report while randomly removing some spaces (p = 0.025)
    report = ""
    for piece in report_pieces:
        report += piece
        if np.random.rand() > 0.025:
            report += " "

    return report


# Define the various parts of a diagnostic report
PREFIXES = [
    "Patient presents with",
    "Individual reports",
    "Case involves",
    "Subject describes"
]

SYMPTOMS = [
    "mild to moderate chest pain",
    "severe headaches",
    "persistent cough",
    "intermittent fever",
    "chronic fatigue",
    "shortness of breath",
    "sudden weight loss"
]

DIAGNOSES = [
    "likely viral infection",
    "possible bacterial infection",
    "early signs of diabetes",
    "indication of chronic fatigue syndrome",
    "symptoms of seasonal allergies",
    "potential thyroid disorder"
]

NOTES = [
    "Advise patient to increase fluid intake.",
    "Suggest dietary changes.",
    "Recommend further testing.",
    "Prescribe rest and over-the-counter medication.",
    "Advise regular exercise.",
    "Schedule follow-up in two weeks."
]
