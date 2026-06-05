#include <bits/stdc++.h>
using namespace std;

bool match_segment(const vector<int>& filtered, int start_pos, const vector<int>& original, int start_ori, int length) {
    if (start_pos + length > (int)filtered.size()) return false;
    for (int i = 0; i < length; i++) {
        if (filtered[start_pos + i] != original[start_ori + i]) return false;
    }
    return true;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N; cin >> N;
    string dummy; cin >> dummy; // "shuffled"
    cin.ignore();
    vector<string> shuffled(N);
    for(int i=0; i<N; ++i) getline(cin, shuffled[i]);
    cin >> dummy; // "original"
    cin.ignore();
    vector<string> original(N);
    for(int i=0; i<N; ++i) getline(cin, original[i]);

    unordered_map<string,int> pos;
    for(int i=0; i<N; ++i) pos[original[i]] = i;

    vector<int> seq(N);
    for(int i=0; i<N; ++i) seq[i] = pos[shuffled[i]];

    const int MAX_MASK = 1 << N;
    vector<int> dp(MAX_MASK, INT_MAX);
    dp[0] = 0;

    for(int mask=0; mask < MAX_MASK; ++mask) {
        if(dp[mask] == INT_MAX) continue;

        int start = 0;
        while(start < N && (mask & (1 << start))) start++;

        for(int end = start; end < N; ++end) {
            bool canTake = true;
            for(int k=start; k<=end; k++) {
                if(mask & (1 << k)) {
                    canTake = false;
                    break;
                }
            }
            if(!canTake) break;

            vector<int> filtered;
            for(int x : seq) {
                if(!(mask & (1 << x))) filtered.push_back(x);
            }

            if(match_segment(filtered, 0, original, start, end - start + 1)) {
                int nmask = mask;
                for(int k = start; k <= end; ++k)
                    nmask |= (1 << k);
                dp[nmask] = min(dp[nmask], dp[mask] + 1);
            }
        }
    }
    cout << dp[MAX_MASK - 1] << "\n";
}
