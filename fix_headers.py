with open('report_skeleton.tex', 'r', encoding='utf-8') as f:
    text = f.read()

disc_header = r"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\section{Discussion (cross-task)}
\label{sec:discussion}

"""
# It currently has \subsection{Visual Cues and Distribution Shift}
text = text.replace(r"\subsection{Visual Cues and Distribution Shift}", disc_header + r"\subsection{Visual Cues and Distribution Shift}")

with open('report_skeleton.tex', 'w', encoding='utf-8') as f:
    f.write(text)
