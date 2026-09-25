import json
import re

with open('results/task2_results.json', 'r') as f:
    task2 = json.load(f)

with open('results/task3_main_results.json', 'r') as f:
    task3 = json.load(f)

with open('results/task4_results.json', 'r') as f:
    task4 = json.load(f)

def fmt(num):
    return f"{num:.2f}"

def fmt_gain(num):
    return f"{num:+.2f}"

# TASK 2: Main Results
t2_main = r"""\begin{table}[H]
    \centering
    \caption{\textbf{Task 2 Benchmark Comparison on PACS (Target: Sketch)}: Source validation metrics per domain, mean source performance, target performance, accuracy gain relative to Source-only, and domain separability score (70/30 logistic regression; 50\% = chance).}
    \label{tab:task2_main_results}
    \footnotesize
    \begin{tabular}{lccccccccc}
        \toprule
        & \multicolumn{3}{c}{\textbf{Source Val Accuracy (\%)}} & \textbf{Mean Source} & \textbf{Mean Source} & \textbf{Target} & \textbf{Target} & \textbf{Target Gain} & \textbf{Domain Sep.} \\
        \cmidrule(lr){2-4}
        \textbf{Method} & \textbf{Photo} & \textbf{Art} & \textbf{Cartoon} & \textbf{Accuracy (\%)} & \textbf{Macro-F1} & \textbf{Acc (\%)} & \textbf{F1} & \textbf{$\Delta$ Acc (\%)} & \textbf{Score (\%)} \\
        \midrule
"""
for m in ["Source-only (ERM)", "DAN", "DANN", "CDAN"]:
    d = task2["main_results"][m]
    m_name = "Source-only" if m == "Source-only (ERM)" else m
    t2_main += f"        {m_name} & {fmt(d['source_val']['Photo'])} & {fmt(d['source_val']['Art'])} & {fmt(d['source_val']['Cartoon'])} & {fmt(d['mean_src_acc'])} & {fmt(d['mean_src_f1'])} & {fmt(d['target_acc'])} & {fmt(d['target_f1'])} & {fmt_gain(d['target_gain'])} & {fmt(d['domain_separability'])} \\\\\n"
t2_main += r"""        \bottomrule
    \end{tabular}
\end{table}"""


# TASK 2: Per Class
t2_pc = r"""\begin{table}[H]
    \centering
    \caption{\textbf{Per-Class Target Accuracy (Sketch)}: Class-level breakdown across adaptation methods to inspect negative transfer.}
    \label{tab:task2_per_class}
    \begin{tabular}{lccccccc}
        \toprule
        \textbf{Method} & \textbf{Dog} & \textbf{Elephant} & \textbf{Giraffe} & \textbf{Guitar} & \textbf{Horse} & \textbf{House} & \textbf{Person} \\
        \midrule
"""
for m in ["Source-only (ERM)", "DAN", "DANN", "CDAN"]:
    d = task2["main_results"][m]["per_class_target"]
    m_name = "Source-only" if m == "Source-only (ERM)" else m
    t2_pc += f"        {m_name} & {fmt(d['dog'])} & {fmt(d['elephant'])} & {fmt(d['giraffe'])} & {fmt(d['guitar'])} & {fmt(d['horse'])} & {fmt(d['house'])} & {fmt(d['person'])} \\\\\n"
t2_pc += r"""        \bottomrule
    \end{tabular}
\end{table}"""


# TASK 2: Controlled Study
t2_cs = r"""\begin{table}[H]
    \centering
    \caption{\textbf{Controlled Alignment Strength Study}: Impact of alignment pressure on source performance, domain separability, and target recognition.}
    \label{tab:task2_design_study}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Hyperparameter Value} & \textbf{Mean Source Acc (\%)} & \textbf{Mean Source F1} & \textbf{Domain Sep. (\%)} & \textbf{Target Acc (\%)} & \textbf{Target F1} \\
        \midrule
"""
for idx, (m, lbl) in enumerate([("DAN (lambda=0.1)", "Setting 1 ($\\lambda=0.1$)"), 
                                ("DAN (lambda=1.0)", "Setting 2 ($\\lambda=1.0$)"), 
                                ("DAN (lambda=10.0)", "Setting 3 ($\\lambda=10.0$)")]):
    d = task2["controlled_study"][m]
    t2_cs += f"        {lbl} & {fmt(d['mean_src_acc'])} & {fmt(d['mean_src_f1'])} & {fmt(d['domain_sep'])} & {fmt(d['target_acc'])} & {fmt(d['target_f1'])} \\\\\n"
t2_cs += r"""        \bottomrule
    \end{tabular}
\end{table}"""


# TASK 3: Main Results
t3_main = r"""\begin{table}[H]
    \centering
    \caption{\textbf{Task 3 Domain Generalization Performance on PACS}: Comparison of ERM, DAN-DG, and SAM across source validation domains, worst-source domain, unseen target (Sketch), and diagnostic metrics.}
    \label{tab:task3_main_results}
    \footnotesize
    \begin{tabular}{lccccccccc}
        \toprule
        & \multicolumn{3}{c}{\textbf{Source Val Acc (\%)}} & \textbf{Mean Source} & \textbf{Worst Source} & \textbf{Sketch Target} & \textbf{Sketch} & \textbf{Source Domain} & \textbf{Sharpness Proxy} \\
        \cmidrule(lr){2-4}
        \textbf{Method} & \textbf{Photo} & \textbf{Art} & \textbf{Cartoon} & \textbf{Acc (\%)} & \textbf{Acc (\%)} & \textbf{Acc (\%)} & \textbf{$\Delta$ Acc (\%)} & \textbf{Sep. Score (\%)} & \textbf{$\Delta_{\text{sharp}}$} \\
        \midrule
"""
for d in task3:
    if d["Method"] in ["ERM", "DAN-DG", "SAM"]:
        t3_main += f"        {d['Method']} & {fmt(d['Photo Acc'])} & {fmt(d['Art Acc'])} & {fmt(d['Cartoon Acc'])} & {fmt(d['Mean Source Acc'])} & {fmt(d['Worst Source Acc'])} & {fmt(d['Sketch Target Acc'])} & {fmt_gain(d['Delta Sketch Acc'])} & {fmt(d['Domain Sep Score'])} & {fmt(d['Sharpness Proxy'])} \\\\\n"
t3_main += r"""        \bottomrule
    \end{tabular}
\end{table}"""


# TASK 3: Per Class
t3_pc = r"""\begin{table}[H]
    \centering
    \caption{\textbf{Per-Class Accuracy on Unseen Sketch}: Comparison between DG methods (Task 3) and UDA methods with target access (Task 2).}
    \label{tab:task3_per_class}
    \begin{tabular}{lccccccc}
        \toprule
        \textbf{Method} & \textbf{Dog} & \textbf{Elephant} & \textbf{Giraffe} & \textbf{Guitar} & \textbf{Horse} & \textbf{House} & \textbf{Person} \\
        \midrule
"""
for d in task3:
    if d["Method"] == "ERM":
        t3_pc += f"        ERM (Shared) & {fmt(d['Target Class Accs']['0'])} & {fmt(d['Target Class Accs']['1'])} & {fmt(d['Target Class Accs']['2'])} & {fmt(d['Target Class Accs']['3'])} & {fmt(d['Target Class Accs']['4'])} & {fmt(d['Target Class Accs']['5'])} & {fmt(d['Target Class Accs']['6'])} \\\\\n"
    elif d["Method"] == "DAN-DG":
        t3_pc += f"        DAN-DG (No Target) & {fmt(d['Target Class Accs']['0'])} & {fmt(d['Target Class Accs']['1'])} & {fmt(d['Target Class Accs']['2'])} & {fmt(d['Target Class Accs']['3'])} & {fmt(d['Target Class Accs']['4'])} & {fmt(d['Target Class Accs']['5'])} & {fmt(d['Target Class Accs']['6'])} \\\\\n"
        # Insert DAN (UDA) from task 2
        d2 = task2["main_results"]["DAN"]["per_class_target"]
        t3_pc += f"        DAN (UDA with Target) & {fmt(d2['dog'])} & {fmt(d2['elephant'])} & {fmt(d2['giraffe'])} & {fmt(d2['guitar'])} & {fmt(d2['horse'])} & {fmt(d2['house'])} & {fmt(d2['person'])} \\\\\n"
    elif d["Method"] == "SAM":
        t3_pc += f"        SAM (No Target) & {fmt(d['Target Class Accs']['0'])} & {fmt(d['Target Class Accs']['1'])} & {fmt(d['Target Class Accs']['2'])} & {fmt(d['Target Class Accs']['3'])} & {fmt(d['Target Class Accs']['4'])} & {fmt(d['Target Class Accs']['5'])} & {fmt(d['Target Class Accs']['6'])} \\\\\n"
t3_pc += r"""        \bottomrule
    \end{tabular}
\end{table}"""


# TASK 3: Controlled Study
t3_cs = r"""\begin{table}[H]
    \centering
    \caption{\textbf{Controlled DG Study (SAM)}: Effect of varying perturbation radius ($\rho$) on source performance, diagnostics, and unseen Sketch recognition.}
    \label{tab:task3_design_study}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Hyperparameter Value} & \textbf{Mean Source Acc (\%)} & \textbf{Worst Source Acc (\%)} & \textbf{Diagnostic Metric} & \textbf{Sketch Acc (\%)} & \textbf{Sketch F1} \\
        \midrule
"""
for d in task3:
    if d["Method"] == "SAM (rho=0.01)":
        t3_cs += f"        $\\rho = 0.01$ & {fmt(d['Mean Source Acc'])} & {fmt(d['Worst Source Acc'])} & {fmt(d['Sharpness Proxy'])} & {fmt(d['Sketch Target Acc'])} & {fmt(d['Sketch Target F1'])} \\\\\n"
    elif d["Method"] == "SAM":
        t3_cs += f"        $\\rho = 0.05$ (Nominal) & {fmt(d['Mean Source Acc'])} & {fmt(d['Worst Source Acc'])} & {fmt(d['Sharpness Proxy'])} & {fmt(d['Sketch Target Acc'])} & {fmt(d['Sketch Target F1'])} \\\\\n"
    elif d["Method"] == "SAM (rho=0.1)":
        t3_cs += f"        $\\rho = 0.1$ & {fmt(d['Mean Source Acc'])} & {fmt(d['Worst Source Acc'])} & {fmt(d['Sharpness Proxy'])} & {fmt(d['Sketch Target Acc'])} & {fmt(d['Sketch Target F1'])} \\\\\n"
t3_cs += r"""        \bottomrule
    \end{tabular}
\end{table}"""

# TASK 4: Post-Hoc
t4_ph = r"""\begin{table}[H]
    \centering
    \caption{\textbf{Post-Hoc Novelty Detection Metrics on Vanilla ResNet-18}: AUROC across unknown subsets and rejection performance calibrated at $\tau = 95^{\text{th}}$ percentile of CIFAR-10 validation unknownness.}
    \label{tab:task4_posthoc_scores}
    \begin{tabular}{lcccccc}
        \toprule
        & \multicolumn{3}{c}{\textbf{AUROC (\%)}} & \multicolumn{3}{c}{\textbf{Rejection Performance at 95\% Val TPR}} \\
        \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \textbf{Score} & \textbf{Near} & \textbf{Far} & \textbf{All} & \textbf{CIFAR-10 Test Acc. (\%)} & \textbf{Near Rej. (\%)} & \textbf{Far Rej. (\%)} \\
        \midrule
"""
for d in task4:
    if d["Model"] == "VANILLA":
        t4_ph += f"        {d['Score']} & {fmt(d['AUROC Near'])} & {fmt(d['AUROC Far'])} & {fmt(d['AUROC All'])} & {fmt(d['CSA (%)'])} & {fmt(d['Near Rej. (%)'])} & {fmt(d['Far Rej. (%)'])} \\\\\n"
t4_ph += r"""        \bottomrule
    \end{tabular}
\end{table}"""


# TASK 4: Trained Models
t4_tm = r"""\begin{table}[H]
    \centering
    \caption{\textbf{Trained-Model OSR Benchmark}: Comparison of Vanilla, GCSC, and PROSER on Closed-Set Accuracy (CSA), AUROC, and validation-calibrated rejection rates at 95\% TPR.}
    \label{tab:task4_trained_models}
    \footnotesize
    \begin{tabular}{lccccccc}
        \toprule
        & & \multicolumn{3}{c}{\textbf{AUROC (\%)}} & \multicolumn{3}{c}{\textbf{Rejection Performance at 95\% Val TPR}} \\
        \cmidrule(lr){3-5} \cmidrule(lr){6-8}
        \textbf{Model / Score} & \textbf{CSA (\%)} & \textbf{Near} & \textbf{Far} & \textbf{All} & \textbf{FPR@95TPR (\%)} & \textbf{Near Rej. (\%)} & \textbf{Far Rej. (\%)} \\
        \midrule
"""
for d in task4:
    if d["Model"] == "VANILLA" and d["Score"] == "MLS":
        t4_tm += f"        Vanilla (MLS) & {fmt(d['CSA (%)'])} & {fmt(d['AUROC Near'])} & {fmt(d['AUROC Far'])} & {fmt(d['AUROC All'])} & {fmt(d['FPR@95TPR (%)'])} & {fmt(d['Near Rej. (%)'])} & {fmt(d['Far Rej. (%)'])} \\\\\n"
    elif d["Model"] == "GCSC" and d["Score"] == "MLS":
        t4_tm += f"        GCSC (MLS) & {fmt(d['CSA (%)'])} & {fmt(d['AUROC Near'])} & {fmt(d['AUROC Far'])} & {fmt(d['AUROC All'])} & {fmt(d['FPR@95TPR (%)'])} & {fmt(d['Near Rej. (%)'])} & {fmt(d['Far Rej. (%)'])} \\\\\n"
    elif d["Model"] == "PROSER" and d["Score"] == "MLS":
        t4_tm += f"        PROSER (MLS) & {fmt(d['CSA (%)'])} & {fmt(d['AUROC Near'])} & {fmt(d['AUROC Far'])} & {fmt(d['AUROC All'])} & {fmt(d['FPR@95TPR (%)'])} & {fmt(d['Near Rej. (%)'])} & {fmt(d['Far Rej. (%)'])} \\\\\n"
    elif d["Model"] == "PROSER" and d["Score"] == "PROSER":
        t4_tm += f"        PROSER (Score) & {fmt(d['CSA (%)'])} & {fmt(d['AUROC Near'])} & {fmt(d['AUROC Far'])} & {fmt(d['AUROC All'])} & {fmt(d['FPR@95TPR (%)'])} & {fmt(d['Near Rej. (%)'])} & {fmt(d['Far Rej. (%)'])} \\\\\n"
t4_tm += r"""        \bottomrule
    \end{tabular}
\end{table}"""


# Replace in report_skeleton.tex
with open('report_skeleton.tex', 'r', encoding='utf-8') as f:
    text = f.read()

def repl(pattern, replacement, text):
    return re.sub(pattern, replacement, text, flags=re.DOTALL)

text = repl(r'\\begin\{table\}\[H\].*?\\caption\{\\textbf\{Task 2 Benchmark Comparison on PACS.*?\\end\{table\}', lambda m: t2_main, text)
text = repl(r'\\begin\{table\}\[H\].*?\\caption\{\\textbf\{Per-Class Target Accuracy \(Sketch\)\}: Class-level breakdown across adaptation methods.*?\\end\{table\}', lambda m: t2_pc, text)
text = repl(r'\\begin\{table\}\[H\].*?\\caption\{\\textbf\{Controlled Alignment Strength Study\}.*?\\end\{table\}', lambda m: t2_cs, text)

text = repl(r'\\begin\{table\}\[H\].*?\\caption\{\\textbf\{Task 3 Domain Generalization Performance on PACS.*?\\end\{table\}', lambda m: t3_main, text)
text = repl(r'\\begin\{table\}\[H\].*?\\caption\{\\textbf\{Per-Class Accuracy on Unseen Sketch\}: Comparison between DG methods \(Task 3\).*?\\end\{table\}', lambda m: t3_pc, text)
text = repl(r'\\begin\{table\}\[H\].*?\\caption\{\\textbf\{Controlled DG Study \(SAM\)\}.*?\\end\{table\}', lambda m: t3_cs, text)

text = repl(r'\\begin\{table\}\[H\].*?\\caption\{\\textbf\{Post-Hoc Novelty Detection Metrics on Vanilla ResNet-18\}.*?\\end\{table\}', lambda m: t4_ph, text)
text = repl(r'\\begin\{table\}\[H\].*?\\caption\{\\textbf\{Trained-Model OSR Benchmark\}.*?\\end\{table\}', lambda m: t4_tm, text)


with open('report_skeleton.tex', 'w', encoding='utf-8') as f:
    f.write(text)
