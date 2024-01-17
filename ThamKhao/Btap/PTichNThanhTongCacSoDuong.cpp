#include <iostream>
using namespace std;
const int MAX = 100;
int a[MAX], n;

void print(int k){
	for(int i = 1; i < k; i++){
		cout << a[i] << "+";
	}
	cout << a[k] << endl;
}

void sinh(int k, int sum, int stt) {
	if (sum > n) { return; }
	
	if (sum == n) { 
		cout << n << " = ";
		print(k-1); 
		return; 
	}
	
	for (int i = stt ; i >= 1; i--) {
		a[k] = i;
		sinh(k + 1, sum + i, i);
	}
}

int main() {
	cout << "Nhap n = "; 
	cin >> n;
	sinh(1, 0, n);
}

