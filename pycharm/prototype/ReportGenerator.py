import os
import shutil
import time
import subprocess

from ReportGeneratorHelper import ReportGeneratorHelper
from Helper import print_warning


# TODO: write latex to file, check for latex compiler (try, catch, if not don't generate reports and print an error -> maybe do this in beginning of CSMS?), compile, put into reports-dir., remove tmp-dir.
#  -> content of file: prepro steps (with before - after figures?), input-parameter, metadata (with figures if interesting (like corrs.)), hardware, iteration-table, final decision & time, ... ?


def generate_parameters_string(parameters):
    hyper_parameters_string = ""
    for (key, value) in parameters.items():
        hyper_parameters_string += f"{key} = {value}, "
    hyper_parameters_string = hyper_parameters_string[:-2]
    hyper_parameters_string = hyper_parameters_string.replace("_", "\\_")

    return hyper_parameters_string


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


def report_preprocessing_part_to_latex(report_tex_file, report_statistics):
    report_tex_file.write("\\section*{Preprocessing} ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part5)
    report_tex_file.write(f"Value & {report_statistics['rows_with_missings']} & {report_statistics['ohe_columns']} & {report_statistics['n_quaniles']} & {report_statistics['non_distinct_rows']} \\\\ ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part6)


def report_setup_part_to_latex(report_tex_file, report_statistics, supervised):
    report_tex_file.write("\\section*{Setup} \\subsection*{Hardware} ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part7)
    report_tex_file.write(f"Value & {report_statistics['hardware']['ram']} & {report_statistics['hardware']['cpu_threads']} & {report_statistics['hardware']['cpu_cores']} \\\\ ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part8)

    report_tex_file.write("\\subsection*{Input Parameters} ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part9)
    report_tex_file.write(f"Value & {report_statistics['parameters']['system_parameters']['accuracy_efficiency_preference']} & {report_statistics['parameters']['system_parameters']['prefer_finding_arbitrary_cluster_shapes']} & {report_statistics['parameters']['system_parameters']['avoid_high_effort_of_hyper_parameter_tuning']} \\\\ ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part10)
    report_tex_file.write(ReportGeneratorHelper.large_latex_part11)
    report_tex_file.write(f"Value & {report_statistics['parameters']['system_parameter_preferences_distance']['find_compact_or_isolated_clusters']} & {report_statistics['parameters']['system_parameter_preferences_distance']['ignore_magnitude_and_rotation']} & {report_statistics['parameters']['system_parameter_preferences_distance']['measure_distribution_differences']} & {report_statistics['parameters']['system_parameter_preferences_distance']['grid_based_distance']} \\\\ ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part12)

    report_tex_file.write("\\subsection*{Metadata} ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part13)
    report_tex_file.write(f"Value & {report_statistics['metadata']['n_rows']} & {report_statistics['metadata']['n_features']} & {report_statistics['metadata']['n_classes']} & {report_statistics['metadata']['nmissing_values']} \\\\ ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part14)
    report_tex_file.write(ReportGeneratorHelper.large_latex_part15)
    report_tex_file.write(f"Value & {report_statistics['metadata']['outlier_percentage']} & {report_statistics['metadata']['high_correlation_percentage']} & {report_statistics['metadata']['class_std_deviation'] if supervised else '-'} \\\\ ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part16)


def report_selection_part_to_latex(report_tex_file, report_statistics, supervised):
    report_tex_file.write("\\section*{Selection Steps} ")
    report_tex_file.write("\\begin{table}[H] \\caption{Listing of all CSMS Iterations} \\adjustbox{max width=\\columnwidth}{ \\begin{tabular}{p{2cm}C{3.5cm}C{2cm}C{7cm}C{3.5cm}} \\toprule \\emph{Iteration} & Selected Algorithm & Selection-Score & Tuned (Hyper-) Parameters & " + ("Accuracy" if supervised else "Silh. Score") + " of Sampling \\\\ \\midrule \\midrule ")

    for i, statistic in enumerate(report_statistics):
        report_tex_file.write(f"Iteration {i + 1} & {statistic['algorithm']} & {statistic['selected_algorithm_score']:.2f} & {generate_parameters_string(statistic['parameters'])} & { statistic['results']['accuracy'] if supervised else statistic['results']['silhouette_score_standardized']:.2f} \\\\ ")

    report_tex_file.write(ReportGeneratorHelper.large_latex_part18)


def report_results_part_to_latex(report_tex_file, learning_type, dataset, result, class_column, report_statistics, supervised):
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

    hyper_parameters_string = generate_parameters_string(report_statistics['parameters'])
    report_statistics['algorithm'] = report_statistics['algorithm'].replace("_", "\\_")

    report_tex_file.write("\\begin{table}[ht!] \\caption{Final Clustering Result} \\centering \\adjustbox{max width=\\columnwidth} { \\begin{tabular}{p{3.5cm}C{5cm}C{5cm}C{5cm}} \\toprule Algorithm & Tuned (Hyper-) Parameters & Reached " + ('Accuracy' if supervised else 'Silh. Score') + " & Total CPU-Runtime of the CSMS \\\\ \\midrule ")
    report_tex_file.write(f"{report_statistics['algorithm']} & {hyper_parameters_string} & {report_statistics['end_results']['accuracy'] if supervised else report_statistics['end_results']['silhouette_score_standardized']:.2f} & {report_statistics['total_cputime']:.2f}s \\\\ ")
    report_tex_file.write(ReportGeneratorHelper.large_latex_part17)


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

    shutil.rmtree("reports/tmp_generate_latex")

    shutil.move("report.pdf", "reports/report.pdf")
    os.rename("reports/report.pdf", f"reports/report_{learning_type}_{dataset_path.split('/')[len(dataset_path.split('/')) - 1]}.pdf")
