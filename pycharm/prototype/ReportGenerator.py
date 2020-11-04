import os
import shutil


def generate_report():
    try:
        os.mkdir("reports/tmp_generate_latex")
    except FileExistsError:
        shutil.rmtree("reports/tmp_generate_latex")
        os.mkdir("reports/tmp_generate_latex")

    new_report_tex_file = open("reports/report.tex", "w+")

    # TODO: write latex to file, check for latex compiler (try, catch, if not don't generate report and print an error -> maybe do this in beginning of CSMS?), compile, put into reports-dir., remove tmp-dir.
    #  -> content of file: prepro steps (with before - after figures?), input-parameter, metadata (with figures if interesting (like corrs.)), hardware, iteration-table, final decision & time, ... ?