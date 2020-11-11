import os
import shutil
import time
import subprocess

from ReportGeneratorHelper import ReportGeneratorHelper
from Helper import print_warning


# TODO: write latex to file, check for latex compiler (try, catch, if not don't generate reports and print an error -> maybe do this in beginning of CSMS?), compile, put into reports-dir., remove tmp-dir.
#  -> content of file: prepro steps (with before - after figures?), input-parameter, metadata (with figures if interesting (like corrs.)), hardware, iteration-table, final decision & time, ... ?


def initialize_report_generation(dataset_path, learning_type):
    try:
        os.mkdir("reports/tmp_generate_latex")
    except FileExistsError:
        shutil.rmtree("reports/tmp_generate_latex")
        os.mkdir("reports/tmp_generate_latex")

    new_report_tex_file = open("reports/tmp_generate_latex/report.tex", "w+")

    new_report_tex_file.write(ReportGeneratorHelper.large_latex_part1)
    new_report_tex_file.write(f"\Large {dataset_path.split('/')[len(dataset_path.split('/')) - 1]} -- {'supervised' if learning_type == 'supervised' else 'unsupervised'} \\\\ ")
    new_report_tex_file.write(ReportGeneratorHelper.large_latex_part2)

    return new_report_tex_file


def report_preprocessing_part_to_latex(report_tex_file):
    report_tex_file.write("\\section*{Preprocessing} ")


def report_setup_part_to_latex(report_tex_file):
    report_tex_file.write("\\section*{Setup} \\subsection*{Hardware} ")

    # TODO

    report_tex_file.write("\\subsection*{Input Parameters} ")

    # TODO

    report_tex_file.write("\\subsection*{Metadata} ")

    # TODO


def report_selection_part_to_latex(report_tex_file):
    report_tex_file.write("\\section*{Selection Steps} ")


def report_results_part_to_latex(report_tex_file, learning_type, dataset, result, class_column):
    report_tex_file.write("\\section*{Results} ")

    if learning_type == "unsupervised":
        if class_column in dataset.columns:
            dataset.drop(columns=[class_column], inplace=True)

        unsupervised_results_figure_filename = "unsupervised_results"

        if len(dataset.columns) == 2:
            ReportGeneratorHelper().save_unsupervised_results_figure_2D(dataset, result["result_labels"], unsupervised_results_figure_filename)
            report_tex_file.write(ReportGeneratorHelper().large_latex_part4)

        if len(dataset.columns) == 3:
            ReportGeneratorHelper().save_unsupervised_results_figure_3D(dataset, result["result_labels"], unsupervised_results_figure_filename + "1", False)
            ReportGeneratorHelper().save_unsupervised_results_figure_3D(dataset, result["result_labels"], unsupervised_results_figure_filename + "2", True)
            report_tex_file.write(ReportGeneratorHelper().large_latex_part3)

    # TODO


def generate_report(report_tex_file, learning_type, dataset_path, verbose=True):
    print("\n-*- Writing results to report ...\n")

    report_tex_file.write("\\end{document}")
    report_tex_file.close()

    try:
        if verbose:
            os.system("pdflatex reports/tmp_generate_latex/report.tex")
        else:
            _ = subprocess.run(["pdflatex", "reports/tmp_generate_latex/report.tex"], stdout=subprocess.DEVNULL)
    except:
        print_warning("<Warning> Could not generate report, maybe pdflatex needs to be installed first.")

    os.remove("report.aux")
    os.remove("report.log")

    # shutil.rmtree("reports/tmp_generate_latex")

    shutil.move("report.pdf", "reports/report.pdf")
    os.rename("reports/report.pdf", f"reports/report_{learning_type}_{dataset_path.split('/')[len(dataset_path.split('/')) - 1]}.pdf")
