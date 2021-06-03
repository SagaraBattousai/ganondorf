|Doc Status| |Coverage| |GitHub license| |Python Lint|

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ganondorf
=========

A federated GAN for Medical Image to Image Segmentation







Data References and Usage
=========================


This repository includes data from the Brain-Tumor-Progression collection and the QIN-BRAIN-DSC-MRI collection, both of which have been published by `The Cancer Imaging Archive`. The original data has been modified to combine all the DICM image slices into one Gzipped NIfTI for each series, for the Brain-Tumor-Progression images the files were renamed Pre.... and Post... for the first and second scan respectivly and for QIN-BRAIN-DSC-MRI the files were renamed as Scan and Mask. The images have all been resized to 256 X 256 using Nearest Neighbors interpolation and the number of slices has been reset to 22; the majority of the Images had 22 slices and those with more had slices dropped to match the look of those with only 22 slices. 

Users of this data must abide by the TCIA Data Usage Policy and the `Creative Commons Attribution 3.0 Unported License`_ under which it has been published. Any publications discussing these data should include references to the following:

  Data Citation
    Schmainda KM, Prah M (2018). Data from Brain-Tumor-Progression. The Cancer Imaging Archive. http://doi.org/10.7937/K9/TCIA.2018.15quzvnb

    Kathleen M Schmainda, Melissa A Prah, Jennifer M Connelly, Scott D Rand. (2016). Glioma DSC-MRI Perfusion Data with Standard Imaging and ROIs [ Dataset ] . The Cancer Imaging Archive. DOI: 10.7937/K9/TCIA.2016.5DI84Js8

  Publication Citation
    Schmainda KM, Prah MA, Rand SD, Liu Y, Logan B, Muzi M, Rane SD, Da X, Yen YF, Kalpathy-Cramer J, Chenevert TL, Hoff B, Ross B, Cao Y, Aryal MP, Erickson B, Korfiatis P, Dondlinger T, Bell L, Hu L, Kinahan PE, Quarles CC. (2018). Multisite Concordance of DSC-MRI Analysis for Brain Tumors: Results of a National Cancer Institute Quantitative Imaging Network Collaborative Project. American Journal of Neuroradiology, 39(6), 1008–1016. DOI: 10.3174/ajnr.a5675


  TCIA Citation
    Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. https://doi.org/10.1007/s10278-013-9622-7


.. _`The Cancer Imaging Archive`: https://cancerimagingarchive.net
.. _`Creative Commons Attribution 3.0 Unported License`: https://creativecommons.org/licenses/by/3.0/



.. |Doc Status| image:: https://readthedocs.org/projects/ganondorf/badge/?version=latest
  :target: https://ganondorf.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

.. |Coverage| image:: https://codecov.io/gh/SagaraBattousai/ganondorf/branch/master/graph/badge.svg?token=8V9SGTAQLM
  :target: https://codecov.io/gh/SagaraBattousai/ganondorf
  :alt: Code Coverage Status

.. |Python Testing| image:: https://github.com/SagaraBattousai/ganondorf/workflows/Python%20Testing/badge.svg
  :target: https://github.com/SagaraBattousai/ganondorf/actions?query=workflow%3APython%20Testing"><img
  :alt: GitHub Actions Python Code Testing

..
  .. |Ganondorf Build| image:: https://github.com/SagaraBattousai/ganondorf/workflows/Python%20Testing/badge.svg
    :target: https://github.com/SagaraBattousai/ganondorf/actions?query=workflow%3APython%20Testing"><img
    :alt: GitHub Actions Python Code Testing
..
  .. |Python Wheel| image:: https://github.com/SagaraBattousai/ganondorf/workflows/Python%20Wheel/badge.svg
    :target: https://github.com/SagaraBattousai/ganondorf/actions?query=workflow%3APython%20Wheel"><img
    :alt: GitHub Actions Python Wheel Package Building

.. |Python Lint| image:: https://github.com/SagaraBattousai/ganondorf/workflows/Python%20Lint/badge.svg
  :target: https://github.com/SagaraBattousai/ganondorf/actions?query=workflow%3APython%20Lint"><img
  :alt: GitHub Actions Python Code Linting













.. |GitHub license| image:: https://img.shields.io/github/license/sagaraBattousai/ganondorf?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI2NCIgaGVpZ2h0PSI2NCI+PHBhdGggZD0iTTMyIDUuN0MxNC4zMzQgNS43LjAxMiAyMC4wMi4wMTIgMzcuNjg2YzAgMTAuOTAzIDUuNDYgMjAuNTI1IDEzLjc4OCAyNi4zaDUuNTkyYTIuMDYgMi4wNiAwIDAxLS4yMDEtLjI5M2MtLjIyNi0uNDQzLS4yMzQtLjk4Mi0uMDI4LTEuOTc1LjE5Ny0uOTU0LjEzNC0xLjkyMy0uMTc1LTIuNjk2bC0uMjI4LS41Ny0uNjAxLjAzMWMtLjc1NC4wNC0xLjIxLS4yMDYtMS40NzgtLjc5NC0uMzMtLjcyNi0uMTg1LTEuMzE1LjQ2NC0xLjkwMS4yMTMtLjE5Mi40MjYtLjM2Ni40NzMtLjM4Ni4wNDctLjAyLjI3LS4yNjEuNDk2LS41MzZsLjQxLS41LjAwMi0uOTQ3Yy4wMDItLjk1OC4xMi0yLjIuNDc2LTQuOTg3LjE5LTEuNDkuMTg3LTEuOTg1LS4wMTUtMi42NDlsLS4wOTgtLjMyMi0uMTcuMjczYy0uMDkzLjE1LS4yODYuMzk5LS40My41NTEtLjIzNy4yNTItLjMwNi4yNzMtLjc3OC4yMzMtLjU4My0uMDQ4LTEuMTA4LS4zNzctMS4zNTEtLjg0Ny0uMjQ3LS40NzctLjE4Ny0xLjQwNC4xMzItMi4wNzguNDY1LS45ODQgMS42MzctMS43ODQgMi42MDItMS43NzkuNDY3LjAwMyAxLjE0LjMxNiAxLjU2LjcyNWwuMzQuMzMuMjc3LS44MTJjLjI0OC0uNzMuMjY0LS44NTguMTY2LTEuMjQ3YTQuMDM3IDQuMDM3IDAgMDEtLjA4My0uNDY3Yy0uMzU0LS4xNDUtMS4zOTgtMS41ODMtMS42MzMtMS44MzMtLjMzOS0uMzU3LS44OTEtMS4xMzItMS4yMjYtMS43MjEtLjU1LS45NjctLjYxNy0xLjI0NC0uNjkyLTIuODYtLjA0NS0uOTg1LS4xNjMtMS45MjYtLjI2Mi0yLjA5Mi0uMzU2LS42LTEuMzEyLTQuMTUtMS40MzEtNS4zMTZsLS4xMjEtMS4xOTUtMy40Ni0uMTE3Yy0zLjY3Mi0uMTI0LTQuMDA1LS4xOTgtNC42LTEuMDE3LS40MDEtLjU1LS4yMjItLjcwNiAxLjUwMy0xLjMxMy43MS0uMjUgMS42MDgtLjY4IDEuOTkyLS45NTMuMzg1LS4yNzQgMi44NDUtMi40IDUuNDIzLTMuMDU0YTEwLjgxMiAxMC44MTIgMCAwMDIuNTktMS4wMzNjLS40MDctLjgxNy0uNTMtMS4xNDgtLjQ4Ni0xLjMyNS4xOTItLjc2Ny44MzYtLjc2MSAyLjM2OC4xNCAxLjIwMy0uODA3IDIuMjc0LTEuNDk0IDMuNzE2LTEuNDkzIDEuMDk0LjMyIDEuNSAxLjMwNSAyLjI5MiAxLjguMjMzLjE0Mi44NjYuNTY2IDEuMzA2LjkxNy0uMDE1LjAxMy0uMDUyLjAzMi0uMDcuMDQ2IDIuMjg4LS44NzcgNS4wOTItMS4yMjQgNy4zNTItLjgxNi42MjIuMTEyIDEuNDY4LjMxIDEuODguNDQxLjg5LjI4My44ODcuMjg0IDEuNzMtLjY4NSAxLjc2MS0yLjAyIDIuMTQ4LTIuNDU0IDIuNjc3LTMuMDAyLjM3Ny0uMzkuNTk4LS41Ny43NzQtLjU4Ni4xMDUtLjAxLjE5NC4wMzguMjkuMTM1LjMwMS4zLjEgMS41MjMtLjY2IDQuMDM4LS4yNC43OTMtLjQyMyAxLjQ1Ni0uNDA3IDEuNDczLjAxNy4wMTcuMjE4LS4wMzUuNDQ1LS4xMTUuNTc4LS4yMDQuOTgxLS4xODIgMS4zNjEuMDc0LjQ4NS4zMjcuNTY4LjU3Mi41NyAxLjY5N2ExMy40NDIgMTMuNDQyIDAgMDEtLjYzMiA0LjA0Yy0uMTgzLjU1Ny0uMzMxIDEuMDktLjMzMSAxLjE4NiAwIC4zMDMuMTEuNDc3LjMwNS40NzcuODMgMCAuOTg4IDEuMTUuMzEgMi4yNS0uNDY4Ljc1OC0uNTE2IDEuMTc1LS4xOTggMS43MTguNTQuOTIxLjk1NiAyLjEyNSAxLjAwNiAyLjkxMy4wNDYuNzE2LjA2Ljc1My4yNTMuNjkyLjk4My0uMzEzIDEuMjIyLS4zNDggMi4xNDgtLjMxMiAxLjE1Ny4wNDUgMS43NjUuMjg4IDIuNDA5Ljk1OS42MjIuNjQ5IDEuMjU3IDEuNjU4IDEuMzg5IDIuMjA3LjE1My42NDEuMDY3IDEuNTc4LS4yMDQgMi4yMDgtLjMyNC43NTMtLjgxOCAxLjExLTEuNzY1IDEuMjczLS4xMy4wMjMtLjM0NS4zODctLjc0MSAxLjI1Mi0uOTM1IDIuMDQzLTEuOTU0IDMuNTY4LTMuNjM0IDUuNDRhODQuNzEgODQuNzEgMCAwMC0xLjEyNiAxLjI3NmwtLjI4MS4zMzMuMjYuNzg1Yy4xNDIuNDMyLjQwMiAxLjQzNi41NzcgMi4yMzJsLjMxOSAxLjQ0Ny4zMy4wMjJjLjE4MS4wMTMuNjE5LjA3Ljk3Mi4xMjguNTg1LjA5Ni42NzYuMDg3IDEuMDUtLjEwNC40ODYtLjI0OCAxLjA4Ny0uMjAyIDEuNDg3LjExMi4yMTIuMTY3LjI5NS4xODEuNTUxLjA5Mi41MTYtLjE4Ljg5MS0uMTI3IDEuMzI4LjE5LjQxOS4zMDMuNjgyLjc2Ni42ODIgMS4xOTkgMCAuMTcuMDc2LjI4LjI0My4zNTcuNjcyLjMwNi44NjEgMS41OTguMzUyIDIuMzk4LS4zMTIuNDkxLS40NjguNjM0LS43NC42ODUtLjE4LjAzNC0uMTkxLjEwMS0uMTY1Ljk1MS4wMTUuNTAzLjA3NSAxLjQwMi4xMzEgMS45OTYuMDc4LjgxMi4wNzUgMS4yLS4wMTEgMS41NTktLjEzNi41NjctLjU1MiAxLjA3Mi0xLjE0NCAxLjQ0N2guNjk2YzguMzI4LTUuNzc1IDEzLjc4Ni0xNS4zOTcgMTMuNzg3LTI2LjNDNjMuOTg4IDIwLjAyMSA0OS42NjcgNS43IDMyIDUuN3oiIGZpbGw9IiM0MzFiMWIiIHN0cm9rZT0iIzAwMCIgc3Ryb2tlLXdpZHRoPSIuMDI1Ii8+PC9zdmc+
