
from typing import List

def compute_L_half() -> List[float]:
  a = [25.3245, 21.5473]
  b = [3.32435, 3.82987]
  c = [0.16801, 0.19823]
  d = [-1.27393, -1.97760]
  e = [-4.85967, -4.32054]

  def L(a, b, c, d, e, x):
    return a / (1.0 + b * pow(x, c)) + d * x + e

  return [L(a[0], b[0], c[0], d[0], e[0], 0.5), L(a[1], b[1], c[1], d[1], e[1], 0.5)]

def main():
  L_halfs = compute_L_half()
  print("L(0.5)    : {}".format(L_halfs))
  print("2 * L(0.5): {}".format([2.0 * L_halfs[0], 2.0 * L_halfs[1]]))

if __name__ == "__main__":
  main()