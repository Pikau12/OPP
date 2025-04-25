#include <iostream>
using namespace std;

int main() {
    cout <<"game"<<endl;
    cout <<"guess the namber from 1 to 10"<<endl;
    cout <<"you have three attempts"<<endl;
    int b=0;

    while (b<3) {
        int a;
        cin >>a;
        if (a == 5) {
            cout <<"win" << endl;
            break;
        } else if (a < 3 || a > 7) {
            cout <<"loose"<< endl;
        } 

        b++;
    }
}