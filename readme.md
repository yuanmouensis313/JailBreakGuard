# JailBreakGuard: Defeating Jailbreak Attacks via Dynamic Prompt Perturbation

This is the official implementation of the paper: *JailBreakGuard: Defeating Jailbreak Attacks via Dynamic Prompt Perturbation*

![项目总体框架图v9](assets/项目总体框架图v9.jpg)

## Introduction

 We propose JailBreakGuard: a jailbreak defense framework, which automatically routes inputs to optimal countermeasures based on text perplexity. Recognizing that single-defense approaches fail against various attack types, we leverage perplexity as a critical indicator to identify attack categories. Character-level perturbation neutralizes high-perplexity attacks, while sentence-level perturbation counters low-perplexity semantic jailbreak. 