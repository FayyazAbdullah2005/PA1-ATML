report_tex = r"""\documentclass{article}

% NeurIPS 2026 Submission Template
% Pass nonatbib if natbib package clashes
\usepackage[preprint]{neurips_2026}

\usepackage[utf8]{inputenc} % allow utf-8 input
\usepackage[T1]{fontenc}    % use 8-bit T1 fonts
\usepackage{hyperref}       % hyperlinks
\usepackage{url}            % simple URL typesetting
\usepackage{booktabs}       % professional-quality tables
\usepackage{amsfonts}       % blackboard math symbols
\usepackage{nicefrac}       % compact symbols for 1/2, etc.
\usepackage{microtype}      % microtypography
\usepackage{xcolor}         % colors
\usepackage{graphicx}       % images
% Fallback macro: embeds real image if file exists, or renders placeholder box if pending
\newcommand{\safeincludegraphics}[2][]{%
  \IfFileExists{#2}{\includegraphics[#1]{#2}}{\fbox{\parbox[c][3.2cm][c]{\linewidth}{\centering \small \texttt{[Figure pending generation: #2]}}}}%
}
\usepackage{amsmath}        % math symbols
\usepackage{subcaption}     % subfigures
\usepackage{float}          % figure positioning

\title{EE-5102 / CS-6304: Advanced Topics in Machine Learning\\Programming Assignment 1: Learning Beyond IID and Closed-Set Assumptions}

\author{%
Abdullah Fayyaz \\
Department of Computer Science / Electrical Engineering \\
Lahore University of Management Sciences \\
\texttt{28100069@lums.edu.pk} \\
\vspace{1mm} \\
}

\begin{document}

\maketitle

\begin{abstract}
\textit{[Abstract summarizing the core findings across all four tasks: inductive biases and visual cue dependencies in deep vision backbones (Task 1), unsupervised domain adaptation under appearance shifts on PACS (Task 2), domain generalization and sharpness-aware optimization without target access (Task 3), and open-set recognition with post-hoc novelty scoring and placeholder learning (Task 4). Code and configuration files to reproduce all experiments are available at: \url{https://github.com/FayyazAbdullah2005/PA1-ATML}]}
\end{abstract}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Introduction and Research Questions}
\label{sec:introduction}

\paragraph{Motivation \& Beyond-IID Framing}
\textit{[Write your introduction discussing the failure modes of standard IID and closed-set assumptions in real-world deployment.]}

\paragraph{Core Research Questions}
\textit{[State the key research questions connecting representations, distribution shifts, and unknown rejection across the tasks.]}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Task 1: Inductive Biases and Feature Representations}
\label{sec:task1}

\subsection{Methodology}
Since the three models have shown high accuracies on large datasets (e.g., ImageNet), we expect to observe high scores on other similar datasets. We measure performance using Top-1 Accuracy, Macro-F1, and Mean Max Confidence. To evaluate the three different models (ResNet-50, ViT-B/16, and OpenCLIP), the STL-10 dataset was used with an 80/20 training/validation split and a seed of 6304. For the three models, the following backbones were frozen and used:
\begin{enumerate}
    \item torchvision ResNet-50 with \texttt{ResNet50\_Weights.IMAGENET1K\_V2}
    \item torchvision ViT-B/16 with \texttt{ViT\_B\_16\_Weights.IMAGENET1K\_V1}
    \item OpenCLIP ViT-B-32 with \texttt{pretrained='openai'}
\end{enumerate}
The linear classifier head for each backbone was then trained using the AdamW optimizer (learning rate $10^{-3}$, weight decay $10^{-4}$, and early stopping patience of 5). Along with these three models, OpenCLIP Zero-Shot classification was also recorded using the prompt \texttt{"a photo of a \{class\}"}.

Since experiments conducted by \citet{geirhos2019imagenet} on texture bias in CNNs showed that grayscaling images did not result in a significant drop in performance for color interventions, we similarly expect little or no difference in performance for ResNet here. Since ViT and OpenCLIP learn global features and are trained on large datasets, we expect these models to also show only small drops in performance. We quantify performance changes using accuracy change ($\Delta \text{Acc}$) and prediction consistency. To test color bias, images were passed through a grayscale filter as the first test and a $180^\circ$ Hue rotation as a second test (evaluated across the 500 clean test images, seed 6304).

\citet{geirhos2019imagenet} found a strong bias towards local textures in CNNs and \citet{raghu2021vision} found that ViTs produce more uniform representations across layers, so we expect CNNs to be more texture-biased and ViTs to be more shape-biased. Evaluated metrics are Shape Bias (\%) and Coverage (\%). A total of 220 cue-conflict images were created using AdaIN style transfer ($\alpha=0.8$) across 5 bidirectional class pairs, filtered by pre-evaluation visual rejection criteria ($\text{SSIM} \ge 0.35$, Sobel edge correlation $\ge 0.40$). Model predictions were categorized into shape, texture, or third-class (other) decisions.

Pixel-level translations of $\delta \in \{0, 8, 16, 32\}$ were applied in four cardinal directions (North, South, East, West) using reflection padding to the 500 test images. The models were evaluated on these images in order to test whether predictions remain stable. Since all three models were trained with random crop augmentations during pretraining, we hypothesize that the models will remain fairly stable under small shifts.

To test the significance of patch structure for each model, each image was divided into a $4 \times 4$ pixel-space grid (yielding 16 square tiles of $56 \times 56$ pixels) and scrambled using a deterministic non-identity patch permutation with seed 6304 across all 500 test images. Top-1 Accuracy was computed along with accuracy change ($\Delta \text{Acc}$) and Consistency scores. Our working hypothesis is that models biased towards local features will perform better in this task (acting as a ``bag of features''), since this intervention destroys global layout while preserving local patch evidence.

In order to examine whether the interventions altered the internal representations even when predictions remained stable, we paired each transformed image with its clean counterpart and measured the cosine stability ($I_T$) of each backbone's representation. We also fit 2D UMAP projections on combined clean and transformed features using seed 6304.

\subsection{Results}

\begin{table}[H]
    \centering
    \caption{\textbf{Clean Baseline Evaluation}: Comparison of linear classifier heads (frozen backbones) and zero-shot OpenCLIP on the balanced 500-image test subset (seed 6304).}
    \label{tab:task1_clean_baseline}
    \begin{tabular}{lccc}
        \toprule
        \textbf{Model / Backbone} & \textbf{Top-1 Accuracy (\%)} & \textbf{Macro-F1} & \textbf{Mean Max Confidence} \\
        \midrule
        ResNet-50 (Linear Head) & 97.40 & 0.9740 & 0.9065 \\
        ViT-B/16 (Linear Head) & 98.00 & 0.9800 & 0.9713 \\
        OpenCLIP ViT-B-32 (Linear Head) & 97.40 & 0.9739 & 0.1397 \\
        OpenCLIP ViT-B-32 (Zero-Shot) & 68.20 & 0.6509 & 0.8977 \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{table}[H]
    \centering
    \caption{\textbf{Impact of Color Interventions}: Top-1 accuracy change ($\Delta$) and prediction consistency relative to clean images.}
    \label{tab:task1_color}
    \begin{tabular}{lcccc}
        \toprule
        & \multicolumn{2}{c}{\textbf{Grayscale}} & \multicolumn{2}{c}{\textbf{180$^\circ$ Hue Rotation}} \\
        \cmidrule(lr){2-3} \cmidrule(lr){4-5}
        \textbf{Model} & \textbf{$\Delta$ Acc (\%)} & \textbf{Consistency (\%)} & \textbf{$\Delta$ Acc (\%)} & \textbf{Consistency (\%)} \\
        \midrule
        ResNet-50 & -2.40 & 96.80 & -1.20 & 97.00 \\
        ViT-B/16 & -1.60 & 96.80 & -1.80 & 96.20 \\
        OpenCLIP ViT-B-32 (Probe) & -2.80 & 95.40 & -3.40 & 94.80 \\
        OpenCLIP ViT-B-32 (Zero-Shot) & -1.40 & 94.60 & -3.20 & 90.40 \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task1/cue_conflict_samples.png}
        \caption{Stylized Cue-Conflict Examples}
        \label{fig:cue_conflict_samples}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task1/rejection_rule_examples.png}
        \caption{Visual Inspection \& Rejection Examples}
        \label{fig:rejection_rule_examples}
    \end{subfigure}
    \caption{\textbf{AdaIN Cue-Conflict Generation}: (a) Bidirectional shape/texture conflict examples across chosen class pairs; (b) Representative accepted vs. rejected stylizations based on defined visual criteria.}
    \label{fig:task1_stylization}
\end{figure}

\begin{table}[H]
    \centering
    \caption{\textbf{Shape vs. Texture Decisions and Bias Metrics}: Decision counts, Shape Bias (\%), and Coverage (\%) across models on 220 valid cue-conflict images.}
    \label{tab:task1_shape_bias}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Model} & \textbf{$N_{\text{shape}}$} & \textbf{$N_{\text{texture}}$} & \textbf{$N_{\text{other}}$} & \textbf{Shape Bias (\%)} & \textbf{Coverage (\%)} \\
        \midrule
        ResNet-50 & 119 & 32 & 69 & 78.81 & 68.64 \\
        ViT-B/16 & 163 & 10 & 47 & 94.22 & 78.64 \\
        OpenCLIP ViT-B-32 (Probe) & 148 & 15 & 57 & 90.80 & 74.09 \\
        OpenCLIP ViT-B-32 (Zero-Shot) & 103 & 32 & 85 & 76.30 & 61.36 \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \safeincludegraphics[width=0.9\linewidth]{figures/task1/cue_conflict_failures.png}
    \caption{\textbf{Qualitative Cue-Conflict Predictions}: Representative examples highlighting model agreements, disagreements (shape preference vs. texture preference), and third-class failure modes.}
    \label{fig:task1_conflict_cases}
\end{figure}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task1/translation_accuracy.png}
        \caption{Top-1 Accuracy vs. Displacement}
        \label{fig:translation_acc}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task1/translation_consistency.png}
        \caption{Prediction Consistency vs. Displacement}
        \label{fig:translation_consistency}
    \end{subfigure}
    \caption{\textbf{Translation Sensitivity Analysis}: (a) Top-1 accuracy and (b) prediction consistency $\text{Consistency}(\delta)$ plotted against spatial pixel displacements $\delta \in \{0, 8, 16, 32\}$ averaged across four cardinal directions.}
    \label{fig:task1_translation_curves}
\end{figure}

\begin{table}[H]
    \centering
    \caption{\textbf{Patch Structure Disruption ($4 \times 4$ Shuffling)}: Performance drop and prediction consistency relative to clean images.}
    \label{tab:task1_patch_shuffle}
    \begin{tabular}{lccc}
        \toprule
        \textbf{Model} & \textbf{Shuffled Top-1 Acc (\%)} & \textbf{$\Delta$ Acc (\%)} & \textbf{Consistency (\%)} \\
        \midrule
        ResNet-50 & 91.40 & -6.00 & 92.00 \\
        ViT-B/16 & 90.80 & -7.20 & 91.00 \\
        OpenCLIP ViT-B-32 (Probe) & 80.60 & -16.80 & 82.00 \\
        OpenCLIP ViT-B-32 (Zero-Shot) & 55.40 & -12.80 & 74.40 \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \safeincludegraphics[width=0.85\linewidth]{figures/task1/patch_shuffle_examples.png}
    \caption{\textbf{Patch-Shuffled Visualizations and Model Predictions}: Examples showing surviving local evidence vs. destroyed global spatial arrangement.}
    \label{fig:task1_patch_examples}
\end{figure}

\begin{table}[H]
    \centering
    \caption{\textbf{Feature Representation Cosine Stability ($I_T$)}: Mean cosine similarity between clean representations and representations under each intervention.}
    \label{tab:task1_cosine_stability}
    \begin{tabular}{lcccc}
        \toprule
        \textbf{Model / Backbone} & \textbf{Grayscale} & \textbf{Cue Conflict} & \textbf{Translation ($\delta=32$)} & \textbf{Patch Shuffling} \\
        \midrule
        ResNet-50 & 0.8029 & 0.3651 & 0.9377 & 0.6953 \\
        ViT-B/16 & 0.7405 & 0.3701 & 0.9473 & 0.6333 \\
        OpenCLIP ViT-B-32 & 0.8891 & 0.7522 & 0.9597 & 0.7560 \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task1/umap_resnet50.png}
        \caption{ResNet-50 Projection}
        \label{fig:umap_resnet}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task1/umap_vitb16.png}
        \caption{ViT-B/16 Projection}
        \label{fig:umap_vit}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task1/umap_clip.png}
        \caption{OpenCLIP ViT-B-32 Projection}
        \label{fig:umap_clip}
    \end{subfigure}
    \caption{\textbf{2D Manifold Visualizations (Clean vs. Transformed)}: 2D UMAP projections of backbone representations combining clean and intervened examples. Points are colored by ground-truth class with circles denoting clean images and crosses denoting transformed images.}
    \label{fig:task1_manifold_projections}
\end{figure}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\section{Task 2: Unsupervised Domain Adaptation (UDA)}
\label{sec:task2}

\subsection{Methodology}
For our UDA experiments, we made use of the PACS dataset, which consists of seven classes across four domains (Painting, Art, Cartoon, Sketch). We used Sketch as our unlabeled target domain. For each source domain, we used an 80/20 training/validation split with a seed of 6304.

For our model, we used a fine-tuned ResNet-18 (\texttt{IMAGENET1K\_V1}) with a re-trained seven-class linear classifier head. The images were resized to 256 x 256 with a random 224 x 224 crop during training and a 224 x 224 center-crop for validation. We froze all BatchNorm running means and variances at their ImageNet values. We used the AdamW optimizer with a learning rate of $10^{-4}$ and weight decay of $10^{-4}$, stopping after five epochs without improvement in the mean source-validation macro-F1.

In order to perform DAN-style alignment, we added a 512-dimensional Maximum Mean Discrepancy (MMD) penalty to the feature space: $L_{DAN} = L_{\text{cls}} + \lambda_{\text{MMD}} \left\| E_s[\phi(F(x_s))] - E_t[\phi(F(x_t))] \right\|_2$, with $\lambda_{MMD}=1$ and a sum of three RBF kernels whose bandwidths were 0.5, 1, and 2 times the median pairwise squared feature distance in the current combined batch.

[Write what we did for Source-only ERM baseline, DANN adversarial alignment with GRL, and the balanced Logistic Regression domain separability diagnostic here.]

[Write what we did for CDAN and class-conditional alignment here: explain the multilinear outer product g(x) = vec(f (x) p), why conditioning on predictions is tested, and how per-class transfer and confusion errors are evaluated.]

[Write what we did for the controlled study here: explain testing lambda\_MMD in \{0.1, 1.0, 10.0\} for DAN, keeping other hyperparameters fixed, and what you hypothesized about increasing alignment strength.]

\subsection{Results}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task2/training_loss_curves.png}
        \caption{Source Classification Loss}
        \label{fig:task2_cls_loss}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task2/alignment_loss_curves.png}
        \caption{Domain Alignment / Discriminator Loss}
        \label{fig:task2_align_loss}
    \end{subfigure}
    \caption{\textbf{Task 2 Training Dynamics}: Convergence curves showing (a) source classification cross-entropy and (b) alignment objectives (MMD penalty / domain discriminator loss) across adaptation epochs.}
    \label{fig:task2_curves}
\end{figure}

\begin{table}[H]
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
        Source-only & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
        DAN & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
        DANN & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
        CDAN & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{table}[H]
    \centering
    \caption{\textbf{Per-Class Target Accuracy (Sketch)}: Class-level breakdown across adaptation methods to inspect negative transfer.}
    \label{tab:task2_per_class}
    \begin{tabular}{lccccccc}
        \toprule
        \textbf{Method} & \textbf{Dog} & \textbf{Elephant} & \textbf{Giraffe} & \textbf{Guitar} & \textbf{Horse} & \textbf{House} & \textbf{Person} \\
        \midrule
        Source-only & -- & -- & -- & -- & -- & -- & -- \\
        DAN & -- & -- & -- & -- & -- & -- & -- \\
        DANN & -- & -- & -- & -- & -- & -- & -- \\
        CDAN & -- & -- & -- & -- & -- & -- & -- \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \safeincludegraphics[width=0.9\linewidth]{figures/task2/confusion_failures.png}
    \caption{\textbf{Target Error Analysis on Sketch}: Dominant confusions and failure cases illustrating semantic preservation vs. negative transfer across methods.}
    \label{fig:task2_confusions}
\end{figure}

\begin{table}[H]
    \centering
    \caption{\textbf{Controlled Alignment Strength Study}: Impact of alignment pressure on source performance, domain separability, and target recognition.}
    \label{tab:task2_design_study}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Hyperparameter Value} & \textbf{Mean Source Acc (\%)} & \textbf{Mean Source F1} & \textbf{Domain Sep. (\%)} & \textbf{Target Acc (\%)} & \textbf{Target F1} \\
        \midrule
        Setting 1 ($\lambda=0.1$) & -- & -- & -- & -- & -- \\
        Setting 2 ($\lambda=1.0$) & -- & -- & -- & -- & -- \\
        Setting 3 ($\lambda=10.0$) & -- & -- & -- & -- & -- \\
        \bottomrule
    \end{tabular}
\end{table}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\section{Task 3: Domain Generalization (DG)}
\label{sec:task3}

\subsection{Methodology}
Explain zero-target-access setup: Sketch is strictly unseen during training, validation, model selection, and hyperparameter tuning. Reusing shared PACS splits, ResNet-18 initialization, and frozen BatchNorm running statistics.

Briefly formulate ERM baseline (reused from Task 2), DAN-DG (pairwise source MMD over (P, A, C), lambda\_DG=1), and SAM (rho=0.05, 2 forward/backward passes).

\subsection{Results}

\begin{table}[H]
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
        ERM & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
        DAN-DG & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
        SAM & -- & -- & -- & -- & -- & -- & -- & -- & -- \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task3/dg_loss_curves.png}
        \caption{Classification \& MMD Penalty Curves}
        \label{fig:task3_loss}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task3/sharpness_vs_generalization.png}
        \caption{Sharpness Proxy vs. Sketch Accuracy}
        \label{fig:task3_sharpness_scatter}
    \end{subfigure}
    \caption{\textbf{Domain Generalization Dynamics and Diagnostics}: (a) Training loss and pairwise MMD penalty trajectories; (b) Empirical correlation between the standardized local sharpness proxy $\Delta_{\text{sharp}}$ and unseen Sketch target performance.}
    \label{fig:task3_diagnostics}
\end{figure}

\begin{table}[H]
    \centering
    \caption{\textbf{Per-Class Accuracy on Unseen Sketch}: Comparison between DG methods (Task 3) and UDA methods with target access (Task 2).}
    \label{tab:task3_per_class}
    \begin{tabular}{lccccccc}
        \toprule
        \textbf{Method} & \textbf{Dog} & \textbf{Elephant} & \textbf{Giraffe} & \textbf{Guitar} & \textbf{Horse} & \textbf{House} & \textbf{Person} \\
        \midrule
        ERM (Shared) & -- & -- & -- & -- & -- & -- & -- \\
        DAN-DG (No Target) & -- & -- & -- & -- & -- & -- & -- \\
        DAN (UDA with Target) & -- & -- & -- & -- & -- & -- & -- \\
        SAM (No Target) & -- & -- & -- & -- & -- & -- & -- \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{table}[H]
    \centering
    \caption{\textbf{Controlled DG Study (SAM)}: Effect of varying perturbation radius ($\rho$) on source performance, diagnostics, and unseen Sketch recognition.}
    \label{tab:task3_design_study}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Hyperparameter Value} & \textbf{Mean Source Acc (\%)} & \textbf{Worst Source Acc (\%)} & \textbf{Diagnostic Metric} & \textbf{Sketch Acc (\%)} & \textbf{Sketch F1} \\
        \midrule
        $\rho = 0.01$ & -- & -- & -- & -- & -- \\
        $\rho = 0.05$ (Nominal) & -- & -- & -- & -- & -- \\
        $\rho = 0.1$ & -- & -- & -- & -- & -- \\
        \bottomrule
    \end{tabular}
\end{table}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\section{Task 4: Open-Set Recognition (OSR)}
\label{sec:task4}

\subsection{Methodology}
Detail CIFAR-10 known classes (90/10 stratified split, seed 6304). Define CIFAR-100 evaluation groups: Near unknowns (bus, pickup\_truck, motorcycle, tractor, wolf, fox, leopard, camel; 800 images) and Far unknowns (bottle, bowl, chair, clock, keyboard, mushroom, sunflower, wardrobe; 800 images).

Describe CIFAR-adapted ResNet-18 (3x3 conv1, stride 1, no max-pooling). Detail vanilla training recipe (SGD, lr=0.1, cosine decay, 100 epochs, batch size 128, seed 6304).

State formulations for MSP, MLS, Energy, and Mahalanobis and calibration details here.

Write GCSC protocol here.

Write PROSER protocol here.

\subsection{Results}
\begin{table}[H]
    \centering
    \caption{\textbf{Post-Hoc Novelty Detection Metrics on Vanilla ResNet-18}: AUROC across unknown subsets and rejection performance calibrated at $\tau = 95^{\text{th}}$ percentile of CIFAR-10 validation unknownness.}
    \label{tab:task4_posthoc_scores}
    \begin{tabular}{lcccccc}
        \toprule
        & \multicolumn{3}{c}{\textbf{AUROC (\%)}} & \multicolumn{3}{c}{\textbf{Rejection Performance at 95\% Val TPR}} \\
        \cmidrule(lr){2-4} \cmidrule(lr){5-7}
        \textbf{Score} & \textbf{Near} & \textbf{Far} & \textbf{All} & \textbf{CIFAR-10 Test Acc. (\%)} & \textbf{Near Rej. (\%)} & \textbf{Far Rej. (\%)} \\
        \midrule
        MSP & -- & -- & -- & -- & -- & -- \\
        MLS & -- & -- & -- & -- & -- & -- \\
        Energy & -- & -- & -- & -- & -- & -- \\
        Mahalanobis & -- & -- & -- & -- & -- & -- \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task4/score_distributions.png}
        \caption{Unknownness Score Distributions}
        \label{fig:task4_score_dist}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task4/roc_curves.png}
        \caption{ROC Curves (Known vs. All Unknowns)}
        \label{fig:task4_roc_curves}
    \end{subfigure}
    \caption{\textbf{Novelty Score Separation}: (a) Density distributions for knowns vs. near and far unknowns; (b) ROC curves across novelty scoring functions.}
    \label{fig:task4_scoring_analysis}
\end{figure}

\begin{table}[H]
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
        Vanilla (MLS) & -- & -- & -- & -- & -- & -- & -- \\
        GCSC (MLS) & -- & -- & -- & -- & -- & -- & -- \\
        PROSER (MLS) & -- & -- & -- & -- & -- & -- & -- \\
        PROSER (Score) & -- & -- & -- & -- & -- & -- & -- \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[H]
    \centering
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task4/near_unknown_failures.png}
        \caption{Accepted Near Unknowns}
        \label{fig:near_failures}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\linewidth}
        \centering
        \safeincludegraphics[width=\linewidth]{figures/task4/far_unknown_failures.png}
        \caption{Accepted Far Unknowns}
        \label{fig:far_failures}
    \end{subfigure}
    \caption{\textbf{Failure Inspection Under Vanilla MLS Threshold $\tau$}: Incorrectly accepted unknown inputs showing unknown class, predicted CIFAR-10 class, logit score, and margin relative to threshold.}
    \label{fig:task4_failures}
\end{figure}

\begin{table}[H]
    \centering
    \caption{\textbf{Case Studies of Accepted Unknown Inputs}: Inspection of three near-unknown and three far-unknown failures under the vanilla MLS decision boundary.}
    \label{tab:task4_failure_cases}
    \begin{tabular}{lccccc}
        \toprule
        \textbf{Type} & \textbf{True CIFAR-100 Class} & \textbf{Predicted CIFAR-10 Class} & \textbf{$u_{\text{MLS}}(x)$} & \textbf{Threshold $\tau$} & \textbf{Failure Characterization} \\
        \midrule
        Near & -- & -- & -- & -- & Plausible semantic overlap \\
        Near & -- & -- & -- & -- & Plausible semantic overlap \\
        Near & -- & -- & -- & -- & Plausible semantic overlap \\
        \midrule
        Far & -- & -- & -- & -- & Surprising feature alias \\
        Far & -- & -- & -- & -- & Surprising feature alias \\
        Far & -- & -- & -- & -- & Surprising feature alias \\
        \bottomrule
    \end{tabular}
\end{table}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newpage
\section{Discussion (cross-task)}
\label{sec:discussion}

\subsection{Task 1.1: Clean Baseline Evaluation}
The results show high accuracies for all three models ($>95\%$) with ViT having the highest accuracy, confidence, and F1-score. ResNet and OpenCLIP (Linear Head) showed very similar accuracy and F1 scores, though ResNet showed higher confidence. OpenCLIP Zero-Shot classification showed significantly lower accuracy at $68.20\%$ and much higher confidence ($0.8977$) compared to OpenCLIP's confidence with a linear probe ($0.1397$). These results are expected, since zero-shot classification is performed using a generic prompt without any fine-tuning.

\subsection{Task 1.2: Chromatic Reliance vs. Shape and Texture Bias}
The results show a mild drop in performance for each model in accuracy and consistency scores. The results are consistent with our original hypothesis with the exception of OpenCLIP. The results suggest that OpenCLIP is the most susceptible to Hue rotation as it showed the largest drops in accuracy ($-3.40\%$ probe, $-3.20\%$ zero-shot) and consistency scores ($90.40\%$ zero-shot). An interpretation of these results is that rotating hue alters the color patterns that OpenCLIP's text and image embeddings rely on, which causes the top-1 prediction to change.

For cue conflicts, the results show that ResNet-50 exhibits relatively lower shape bias ($78.81\%$) compared to OpenCLIP Probe ($90.80\%$) and ViT ($94.22\%$), choosing texture 32 times. ViT shows the highest coverage out of all models at $78.64\%$ followed by OpenCLIP Probe ($74.09\%$) and then ResNet-50 ($68.64\%$), with OpenCLIP Zero-Shot displaying the lowest coverage ($61.36\%$) and lowest shape bias ($76.30\%$). These results are consistent with our original hypothesis and show that OpenCLIP Zero-Shot displays the highest confusion ($N_{\text{other}}=85$) when predicting labels for cue-conflict images. The reason for these results is that CNNs do in fact have a higher texture bias due to local convolutions, while the self-attention mechanism allows ViT to learn global boundaries better. Furthermore, OpenCLIP Probe also exhibits high shape bias due to its Vision Transformer backbone; the lower shape bias and coverage in Zero-Shot can be explained by the models' high confusion when textures conflict with text prompt semantics, as illustrated in Figure~\ref{fig:task1_conflict_cases}.

\subsection{Task 1.3: Spatial Invariance: Translation vs. Patch Permutation}
The results show near-zero changes in accuracy and prediction consistency for all models under translation ($>96.5\%$ consistency even at $\delta=32$). This suggests that spatial displacement has a negligible effect on model accuracy, though OpenCLIP ViT-B/32 showed a slightly larger drop ($97.05\%$ probe consistency) because a 32-pixel shift spans an entire patch stride on its $32 \times 32$ grid.

For patch shuffling, the results in Table~\ref{tab:task1_patch_shuffle} show drops in accuracy across all models, with OpenCLIP Probe showing the largest accuracy drop ($-16.80\%$) and Zero-Shot dropping to $55.40\%$ accuracy ($74.40\%$ consistency). In contrast, ResNet-50 and ViT-B/16 show only modest drops ($-6.00\%$ and $-7.20\%$) while retaining $>90\%$ accuracy and $>91\%$ consistency. This demonstrates that ResNet and ViT function effectively as bag-of-features classifiers: they retain high accuracy when global layout is destroyed because surviving local evidence inside individual $56 \times 56$ tiles is sufficient for classification (Figure~\ref{fig:task1_patch_examples}). OpenCLIP relies much more heavily on global scene composition learned from multimodal web pretraining, causing it to degrade more sharply when spatial arrangement is permuted.

\subsection{Task 1.4: Representation Geometry vs. Prediction Stability}
The results show that all interventions altered representations to varying degrees, revealing a clear mismatch between prediction stability and representation stability. For instance, while Grayscale caused only a minimal accuracy drop in Task 1.2 ($<2.5\%$), cosine stability dropped significantly to $0.7405$ in ViT and $0.8029$ in ResNet, showing that internal representations shifted substantially even though the final classifier output remained unchanged. Translation had the most stable representation ($I_T > 0.93$ across all models), whereas Cue Conflict and Patch Shuffling caused large disruptions, with Cue Conflict collapsing cosine stability to $\approx 0.37$ for ResNet and ViT. In the UMAP projections (Figure~\ref{fig:task1_manifold_projections}), clean and transformed points still cluster by class (explaining why accuracy remains high), but the transformed points are scattered and shifted within the clusters. Comparing OpenCLIP's zero-shot decisions with its trained linear head further confirms this: the trained linear head is able to navigate these representation shifts and achieve $97.40\%$ accuracy, whereas zero-shot cosine similarity with a fixed text prompt struggles under distribution drift.

\subsection{Task 1.5: Architectural Bias vs. Pretraining, Supervision, and Capacity}
Since OpenCLIP and ViT both have Transformer-based architectures, some of their differences can adequately be explained by differences in pretraining. It is reasonable to conclude from our results that OpenCLIP's high cosine stability across interventions ($I_T \ge 0.75$) is likely because it was pretrained with InfoNCE loss over diverse web images, which trains the model to keep matching image and text representations aligned and stable. ViT and OpenCLIP's higher shape bias likely stems from architectural differences between CNNs and Transformers: the evidence for this is that the pretraining for ViT and ResNet is similar (both being trained on ImageNet with cross-entropy loss), yet ViT achieved a much higher shape bias ($94.22\%$ vs. $78.81\%$). Furthermore, ViT's higher coverage ($78.64\%$ vs. $68.64\%$) means that it would be inappropriate to attribute this difference to noise. In terms of supervision and classifier heads, OpenCLIP's low zero-shot performance ($68.20\%$) and high third-class confusion ($N_{\text{other}}=85$) largely disappeared once the linear probe was trained ($97.40\%$ accuracy, $90.80\%$ shape bias), proving this was an artifact of the static prompt bottleneck rather than the underlying representation. Finally, OpenCLIP's greater sensitivity to 32-pixel translation compared to ViT-B/16 reflects the architectural difference in patch token size ($32 \times 32$ vs. $16 \times 16$).

\subsection{Task 2.1: UDA Baseline Domain Gap}
[Address RQ 2.1 here: How large is the source-to-target domain gap for Source-only ERM, and which classes account for the most important failures?]

\subsection{Task 2.2: Domain Separability vs. Target Recognition}
[Address RQ 2.2 here: Across the four methods, does lower domain separability correspond to better target recognition? Use aggregate and class-level evidence to identify successful adaptation or negative transfer.]

\subsection{Task 2.3: Marginal vs. Class-Conditional Alignment}
[Address RQ 2.3 here: How does class-conditional alignment compare with the marginal alignment used by DAN and DANN? Do CDAN's gains or failures support the claim that conditioning helps preserve semantic structure?]

\subsection{Task 2.4: Controlled Alignment Strength Study}
[Address RQ 2.4 here: How does increasing alignment strength change source performance, domain separability, and target performance? What trade-off does your controlled study reveal, and what setting could have been selected without consulting target labels?]

\subsection{Task 3.1 \& 3.2: Source Invariance vs. Unseen Generalization}
[Address RQ 3.1 \& RQ 3.2 here: How well do mean-source and worst-source validation performance predict performance on Sketch? Which class-level failures are not visible from the aggregate source results? Does DAN-DG make the observed source domains less distinguishable, and does that invariance improve Sketch recognition? Identify any evidence that source alignment instead removed class-discriminative information.]

\subsection{Task 3.3: Parameter Stability and the Sharpness Proxy}
[Address RQ 3.3 here: Does SAM reduce the common sharpness proxy relative to ERM and DAN-DG? Explain whether the ranking by local stability agrees with the ranking by Sketch performance.]

\subsection{Task 3.4: Target-Free DG vs. Target-Aware UDA}
[Address RQ 3.4 here: Relative to the shared ERM baseline, what does the comparison between target-aware DAN in Task 2 and target-free DAN-DG in Task 3 suggest about the value of unlabeled Sketch data? What limitations prevent this from being a complete causal attribution?]

\subsection{Task 4.1 \& 4.2: Post-Hoc Novelty Scoring}
[Address RQ 4.1 \& RQ 4.2 here: How does semantic similarity affect rejection? Which near and far classes are most often accepted, which CIFAR-10 labels absorb them, and which failures are semantically plausible? What do MSP, MLS, Energy, and Mahalanobis each capture? Use agreements and disagreements among their scores to explain when confidence, logit magnitude, or feature distance is most informative.]

\subsection{Task 4.3 \& 4.4: Placeholder Learning (GCSC and PROSER)}
[Address RQ 4.3 \& RQ 4.4 here: Do GCSC's random augmentations improve closed-set accuracy, open-set recognition, both, or neither? Explain why its effect differs or remains similar for near and far unknowns. Does PROSER improve rejection beyond the Vanilla and GCSC models, and what CSA trade-off results? Are its gains consistent with classifier and data placeholders tightening the boundaries between known classes, and where does interpolation between known classes remain insufficient?]

\subsection{Task 4: Failure Modes Analysis}
[Write discussion on qualitative failure inspection here.]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\section{Conclusion}
\label{sec:conclusion}

\textit{[A very short conclusion summarizing the key empirical takeaways.]}

\newpage
\bibliographystyle{plain}
\begin{thebibliography}{99}

\bibitem{geirhos2019imagenet}
R.~Geirhos, P.~Rubisch, C.~Michaelis, M.~Bethge, F.~A. Wichmann, and W.~Brendel.
\newblock Image{N}et-trained {CNN}s are biased towards texture; increasing shape bias improves accuracy and robustness.
\newblock In \emph{International Conference on Learning Representations (ICLR)}, 2019.

\bibitem{raghu2021vision}
M.~Raghu, T.~Unterthiner, S.~Kornblith, M.~Zhang, and A.~Dosovitskiy.
\newblock Do vision transformers see like convolutional neural networks?
\newblock In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, volume~34, pages 12111--12121, 2021.

\bibitem{ganin2016domain}
Y.~Ganin, E.~Ustinova, H.~Ajakan, P.~Germain, H.~Larochelle, F.~Laviolette, M.~Marchand, and V.~Lempitsky.
\newblock Domain-adversarial training of neural networks.
\newblock \emph{Journal of Machine Learning Research}, 17(59):1--35, 2016.

\bibitem{long2015learning}
M.~Long, Y.~Cao, J.~Wang, and M.~Jordan.
\newblock Learning transferable features with deep adaptation networks.
\newblock In \emph{International Conference on Machine Learning (ICML)}, pages 97--105, 2015.

\bibitem{long2018conditional}
M.~Long, Z.~Cao, J.~Wang, and M.~I. Jordan.
\newblock Conditional adversarial domain adaptation.
\newblock In \emph{Advances in Neural Information Processing Systems (NeurIPS)}, 2018.

\bibitem{foret2021sharpness}
P.~Foret, A.~Kleiner, H.~Mobahi, and B.~Neyshabur.
\newblock Sharpness-aware minimization for efficiently improving generalization.
\newblock In \emph{International Conference on Learning Representations (ICLR)}, 2021.

\bibitem{vaze2022open}
S.~Vaze, K.~Han, A.~Vedaldi, and A.~Zisserman.
\newblock Open-set recognition: a good closed-set classifier is all you need?
\newblock In \emph{International Conference on Learning Representations (ICLR)}, 2022.

\bibitem{zhou2021learning}
D.-W. Zhou, H.-J. Ye, and D.-C. Zhan.
\newblock Learning placeholders for open-set recognition.
\newblock In \emph{IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, pages 4401--4410, 2021.

\end{thebibliography}

\end{document}
"""

with open('report_skeleton.tex', 'w', encoding='utf-8') as f:
    f.write(report_tex)
