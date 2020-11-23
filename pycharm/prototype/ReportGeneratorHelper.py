import os
import plotly.graph_objects as go
import numpy as np


class ReportGeneratorHelper:
    large_latex_part1 = "\\documentclass{article}[11pt] \\usepackage{booktabs} \\usepackage{float} \\usepackage{array} \\newcolumntype{C}[1]{>{\\centering\\let\\newline\\\\\\arraybackslash\hspace{0pt}}m{#1}} \\usepackage{subcaption} \\usepackage{geometry} \\geometry{a4paper, left=25mm, right=25mm, top=25mm, bottom=25mm} \\usepackage[english]{babel} \\usepackage[utf8]{inputenc} \\usepackage{diagbox} \\usepackage{adjustbox} \\usepackage{multirow} \\usepackage{datetime} \\renewcommand\\familydefault{\sfdefault} \\title{ \\LARGE Clustering Selection Management System Report \\\\ \\vspace{0.2cm} "
    large_latex_part2 = "\\vspace{0.1cm} \\today \\ -- \\currenttime \\\\ } \\author{ } \\date{ } \\begin{document} \\sloppy \\maketitle \\vspace{0.5cm} \\normalsize "
    large_latex_part3 = "\\begin{figure}[h] \\centering \\begin{subfigure}[b]{0.48\\textwidth} \\centering \\includegraphics[width=\\textwidth]{reports/tmp_generate_latex/figures/unsupervised_results1.pdf} \\subcaption{Clustering View 1} \\end{subfigure} \\begin{subfigure}[b]{0.48\\textwidth} \\centering \\includegraphics[width=\\textwidth]{reports/tmp_generate_latex/figures/unsupervised_results2.pdf} \\subcaption{Clustering View 2} \\end{subfigure} \\caption{Final Clustering result, represented in two different views of the same plot} \\end{figure} "
    large_latex_part4 = "\\begin{figure}[h] \\centering \\includegraphics[width=0.6\\textwidth]{reports/tmp_generate_latex/figures/unsupervised_results.pdf} \\caption{Final Clustering result for the 2-dimensional dataset} \\end{figure} "
    large_latex_part5 = "\\begin{table}[ht!] \\caption{Specific Metrics for all Preprocessing steps} \\centering \\adjustbox{max width= \\columnwidth} { \\begin{tabular}{p{1.5cm}||C{4.5cm}C{4.5cm}C{4.5cm}C{4.5cm}} \\toprule \\emph{Metric} & \\# Rows w/ missings removed & \\# Converted columns to OHE & \\# Quantiles for Quant. Scal. & \\# Non Distinct rows removed \\\\ "
    large_latex_part6 = "\\bottomrule \\end{tabular} } \\end{table} "
    large_latex_part7 = "\\begin{table}[ht!] \\caption{Hardware Statistics of the underlying Hardware Setup} \\centering \\adjustbox{max width= \\columnwidth} { \\begin{tabular}{p{1.5cm}||C{5cm}C{4cm}C{4cm}}  \\toprule  \\emph{Statistic} &  \\emph{Amount of main memory} &  \\emph{ \\# CPU-Threads} & \\# CPU-Cores \\\\ "
    large_latex_part8 = "\\bottomrule \\end{tabular} } \\end{table} "
    large_latex_part9 = "\\begin{table}[ht!] \\caption{Given \\emph{General} Input-Parameter Values} \\centering \\adjustbox{max width=\\columnwidth} { \\begin{tabular}{p{1.5cm}||C{5cm}C{6cm}C{7.5cm}} \\toprule \\emph{Parameter} & Accuracy Efficiency Preference & Prefer Finding arbitrary Cluster Shapes? & Avoid High Effort of (Hyper-) Parameter Tuning? \\\\ "
    large_latex_part10 = "\\bottomrule \\end{tabular} } \\end{table} "
    large_latex_part11 = "\\begin{table}[ht!] \\caption{Given \\emph{Distance-Metric-based} Input-Parameter Values} \\centering \\adjustbox{max width=\\columnwidth} { \\begin{tabular}{p{1.5cm}||C{5.5cm}C{5cm}C{5cm}C{4cm}} \\toprule \\emph{Parameter} & Find Compact or Isolated Clusters? & Ignore Magnitude and Rotation? & Measure Distribution Differences? & Grid-based Distance? \\\\ "
    large_latex_part12 = "\\bottomrule \\end{tabular} } \\end{table} "
    large_latex_part13 = "\\begin{table}[ht!] \\caption{ \\emph{General} Profiled Metadata Results regarding the Dataset} \\centering \\adjustbox{max width= \\columnwidth} { \\begin{tabular}{p{1.5cm}||C{2cm}C{2cm}C{2cm}C{3cm}} \\toprule \\emph{Statistic} & \\emph{ \\#Rows} & \\emph{ \\#Columns} & \\#Classes & \\# Missing Values \\\\ "
    large_latex_part14 = "\\bottomrule \\end{tabular} } \\end{table} "
    large_latex_part15 = "\\begin{table}[ht!] \\caption{\\emph{Further} Profiled Metadata Results regarding the Dataset} \\centering \\adjustbox{max width=\\columnwidth} { \\begin{tabular}{p{1.5cm}||C{2cm}C{3cm}C{4cm}} \\toprule \\emph{Statistic} & \\emph{Outlier \\%} & \\emph{High Correlation \\%} & Class Std. Deviation \\\\ "
    large_latex_part16 = "\\bottomrule \\end{tabular} } \\end{table} "
    large_latex_part17 = "\\bottomrule \\end{tabular} } \\end{table} "
    large_latex_part18 = "\\bottomrule \\end{tabular} } \\end{table} "

    def save_unsupervised_results_figure_3D(self, dataset, labels, filename, rotate):
        colors = ["#e63946", "#457b9d", "#1d3557", "#2a9d8f", "#ef476f", "#06d6a0", "#b56576", "#006d77", "#2b2d42", "#52b788",
                  "#0b525b", "#4cc9f0"]

        traces = []
        for i, cluster in enumerate(set(labels)):
            cluster_points = dataset.loc[labels == cluster]

            traces.append(go.Scatter3d(x=np.array(cluster_points)[..., 0], y=np.array(cluster_points)[..., 1],
                                       z=np.array(cluster_points)[..., 2], mode='markers',
                                       name="Cluster " + str(cluster + 1),
                                       marker=dict(size=3, color=colors[cluster % len(colors)])))

        layout = go.Layout()
        figure = go.Figure(data=traces, layout=layout)

        if rotate:
            camera_y = -2
        else:
            camera_y = 2

        camera = dict(eye=dict(x=2, y=camera_y, z=0.1))
        figure.update_layout(height=800, width=800, scene=dict(xaxis_title=dataset.columns[0], yaxis_title=dataset.columns[1], zaxis_title=dataset.columns[2]),
                             font_family="CMU Sans Serif", font_color="#4A494A", font_size=12, plot_bgcolor="#E4F4FA", scene_camera=camera)

        save_directory = "reports/tmp_generate_latex/figures"

        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        figure.write_image(save_directory + f"/{filename}.pdf", scale=3.0)


    def save_unsupervised_results_figure_2D(self, dataset, labels, filename):
        colors = ["#e63946", "#457b9d", "#1d3557", "#2a9d8f", "#ef476f", "#06d6a0", "#b56576", "#006d77", "#2b2d42",
                  "#52b788", "#0b525b", "#4cc9f0"]

        figure = go.Figure()

        for i, cluster in enumerate(set(labels)):
            cluster_points = dataset.loc[labels == cluster]

            figure.add_trace(go.Scatter(x=np.array(cluster_points)[..., 0], y=np.array(cluster_points)[..., 1], mode="markers",
                                        marker_color=colors[i % len(colors)], name="Cluster " + str(cluster + 1)))

        figure.update_xaxes(title_text=dataset.columns[0])
        figure.update_yaxes(title_text=dataset.columns[1])

        figure.update_layout(height=800, width=800, font_family="CMU Sans Serif", font_color="#4A494A", font_size=16, plot_bgcolor="#E4F4FA")
        figure.update_traces(marker=dict(size=4))

        save_directory = "reports/tmp_generate_latex/figures"

        if not os.path.exists(save_directory):
            os.mkdir(save_directory)

        figure.write_image(save_directory + f"/{filename}.pdf", scale=3.0)
