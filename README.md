# Adversarial Evasion Attacks Detection for Tree-Based Ensembles: A Representation Learning Approach

This repository contains the experiment code from the paper.

## Abstract

Research on adversarial evasion attacks primarily focuses on neural network models due to their popularity in fields such as computer vision and natural language processing, as well as their properties that facilitate the search for adversarial examples with minimal input changes. However, due to their high performance and explainability, decision trees and tree ensembles remain widely used, particularly in domains dominated by tabular data. In recent years, several studies have introduced novel adversarial attacks specifically targeting decision trees and tree ensembles, resulting in numerous papers addressing robust versions of these models.
In this research, we aim to develop an adversarial detector for attacks on an ensemble of decision trees. While previous works have demonstrated the generation of more robust tree ensembles, considering evasion attacks during ensemble generation impacted the models' performance. Our approach introduces a method to detect adversarial samples without compromising the structure or original performance of the target model. Our analysis shows that leveraging representation learning based on the tree structure significantly improves detection rates compared to state-of-the-art techniques and training adversarial detectors using the original dataset representation.

## Prerequisites & Setup

## How To Run

## Experimental Results

Showing here two box plots that represent the differences in AUC of ROC and PRC, between our method’s performance and the OC-score method’s.
The distributions of the differences comparing our method to OC-score for L2 norm.

ROC-AUC differences for XGBoost experiments between our method and OC-score for L2 norm:

<center><img src="imgs_with_background/xgboost_roc_diff_per_attack_2.png" alt="" width="700" height="400"></center>


### Figure 2: 

PRC-AUC differences for XGBoost experiments between our method and OC-score for L2 norm:

<center><img src="imgs_with_background/xgboost_pr_diff_per_attack_2.png" alt="" width="700" height="400"></center>

Full results represented <a href="https://galbraun.github.io/trees_adversarial_detector">here</a>.