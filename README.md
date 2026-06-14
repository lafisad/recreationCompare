# recreationCompare

**current version: 0.2.0**  

This project aims to provide a way to easily compare an original to a musical recreation.
For this I am using **matplotlib**, **numpy**, **scipy** and **ffmpeg**.

# Installing

To install you need to add some python packages:

- `matplotlib`
- `numpy`
- `scipy`

*or you can do*

`pip install -r requirements.txt`

To use, this is the syntax:
`python compare.py originalFile recreation`

This *should* work with any audio type that **ffmpeg** supports too.

# Important resources

- [MIT License](LICENSE)
- [CONTRIBUTING.md](CONTRIBUTING.md)

# Testing

To run the tests, you need to have the `pytest` package installed:

`pip install pytest`

Then you can run the tests with:

`pytest`
