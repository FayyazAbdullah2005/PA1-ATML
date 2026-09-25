import re

with open('report_skeleton.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Fix grammar in Task 1 & 2
replacements = {
    "Since the three models have shown high accuracies on large datasets (e.g. ImageNet), we expect to see high scores on other similar datasets.":
    "Since the three models have shown high accuracies on large datasets (e.g., ImageNet), we expect to observe high scores on other similar datasets.",
    
    "To evaluate three different models (ResNet-50, ViT-B/16, and OpenCLIP), the STL-10 dataset was used with an 80/20 training/validation split and \\text{seed}=6304.":
    "To evaluate the three different models (ResNet-50, ViT-B/16, and OpenCLIP), the STL-10 dataset was used with an 80/20 training/validation split and a seed of 6304.",
    
    "The linear classifier head for each backbone was then trained using the AdamW optimizer with (\\text{lr}=10^{-3}, \\text{wd}=10^{-4}, \\text{and early stopping patience} = 5).":
    "The linear classifier head for each backbone was then trained using the AdamW optimizer (learning rate $10^{-3}$, weight decay $10^{-4}$, and early stopping patience of 5).",
    
    "Along with these three models, OpenCLIP Zero-Shot classification was also recorded using the prompt \\texttt{\"a photo of a \\{class\\}.\"}":
    "Along with these three models, OpenCLIP Zero-Shot classification was also recorded using the prompt \\texttt{\"a photo of a \\{class\\}\"}.",
    
    "These results are expected since the Zero-Shot classification is done using a generic prompt without any fine-tuning.":
    "These results are expected, since zero-shot classification is performed using a generic prompt without any fine-tuning.",
    
    "Since in the experiments conducted by \\citet{geirhos2019imagenet} on texture bias in CNNs grayscaling images did not result in a significant drop in performance for color interventions in CNNs, we similarly expect little or no difference in performance for ResNet here.":
    "Since experiments conducted by \\citet{geirhos2019imagenet} on texture bias in CNNs showed that grayscaling images did not result in a significant drop in performance for color interventions, we similarly expect little or no difference in performance for ResNet here.",
    
    "Since ViT and OpenCLIP learn global features and are trained on large datasets, we expect those models to also show only small drops in performance.":
    "Since ViT and OpenCLIP learn global features and are trained on large datasets, we expect these models to also show only small drops in performance.",
    
    "For our UDA experiments we made use of the PACS dataset which consists of seven classes across four domains(Painting,Art,Cartoon,Sketch). We used Sketch as our unlabeled domain. For each source domain we used an 80/20 training/validation split with seed 6304":
    "For our UDA experiments, we made use of the PACS dataset, which consists of seven classes across four domains (Painting, Art, Cartoon, Sketch). We used Sketch as our unlabeled target domain. For each source domain, we used an 80/20 training/validation split with a seed of 6304.",
    
    "For our model we used a fine tuned ResNet-18(IMAGENET1K_V1) with a re-trained  seven-class linear classifier head.":
    "For our model, we used a fine-tuned ResNet-18 (\\texttt{IMAGENET1K\\_V1}) with a re-trained seven-class linear classifier head.",
    
    "We freezed all BatchNorm running means and variance at their ImageNet values.":
    "We froze all BatchNorm running means and variances at their ImageNet values.",
    
    "We used the AdamW optimizer  with learning rate $10^{-4}$ and Weight Decay $10^{-4}$,stopping after five epochs without improvement in mean source-validation macro-F1.":
    "We used the AdamW optimizer with a learning rate of $10^{-4}$ and weight decay of $10^{-4}$, stopping after five epochs without improvement in the mean source-validation macro-F1.",
    
    "In order to perform DAN-style alignment. We added a 512 dimension Maximum Mean discrepancy penalty to the feature space $L_{DAN} = L_{\\text{cls}} + \\lambda_{\\text{MMD}} \\left\\| E_s[\\phi(F(x_s))] - E_t[\\phi(F(x_t))] \\right\\|_2 $ with $\\lambda_{MMD}=1$ and a sum of three RBF kernels of bandwidths were 0.5, 1, and 2\ntimes the median pairwise squared feature distance in the current combined batch.":
    "In order to perform DAN-style alignment, we added a 512-dimensional Maximum Mean Discrepancy (MMD) penalty to the feature space: $L_{DAN} = L_{\\text{cls}} + \\lambda_{\\text{MMD}} \\left\\| E_s[\\phi(F(x_s))] - E_t[\\phi(F(x_t))] \\right\\|_2$, with $\\lambda_{MMD}=1$ and a sum of three RBF kernels whose bandwidths were 0.5, 1, and 2 times the median pairwise squared feature distance in the current combined batch."
}

for k, v in replacements.items():
    text = text.replace(k, v)

# Regex to find Discussion blocks
discussions = []

def repl_discussion(match):
    title = match.group(1).strip()
    content = match.group(2).strip()
    discussions.append((title, content))
    return "" # Remove from current location

text = re.sub(r'\\noindent\s*\\textbf\{(Discussion.*?)\}\s*(.*?)(?=\n\n% -|\n\n% =|\n\n\\subsection|\n\n\\newpage|\n\n\\section|\Z)', repl_discussion, text, flags=re.DOTALL)
text = re.sub(r'\\paragraph\{(Discussion.*?)\}\s*(.*?)(?=\n\n% -|\n\n% =|\n\n\\subsection|\n\n\\newpage|\n\n\\section|\Z)', repl_discussion, text, flags=re.DOTALL)
text = re.sub(r'\\subsubsection\*\{(Discussion.*?)\}\s*(.*?)(?=\n\n% -|\n\n% =|\n\n\\subsection|\n\n\\newpage|\n\n\\section|\Z)', repl_discussion, text, flags=re.DOTALL)

# Reconstruct Section 5 (Synthesis)
new_discussion_text = "\\section{Discussion (cross-task)}\n\\label{sec:discussion}\n\n"
for title, content in discussions:
    clean_title = title.replace(':', '')
    new_discussion_text += f"\\subsection{{{clean_title}}}\n{content}\n\n"

synthesis_pattern = r'\\section\{Synthesis of Results and Cross-Task Discussion\}.*?(?=\\section\{Conclusion\})'
text = re.sub(synthesis_pattern, lambda m: new_discussion_text, text, flags=re.DOTALL)

# Simplify conclusion
conclusion_pattern = r'\\section\{Conclusion\}.*?(?=\\newpage)'
short_conclusion = "\\section{Conclusion}\n\\label{sec:conclusion}\n\n\\textit{[A very short conclusion summarizing the key empirical takeaways.]}\n\n"
text = re.sub(conclusion_pattern, lambda m: short_conclusion, text, flags=re.DOTALL)

with open('report_skeleton.tex', 'w', encoding='utf-8') as f:
    f.write(text)
    
print(f"Extracted {len(discussions)} discussion blocks.")
