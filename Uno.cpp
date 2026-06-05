#include <bits/stdc++.h>
using namespace std;

// Find representative of a set using path compression (for union-find)
int findParent(int playerIndex, vector<int>& parent) {
    if (parent[playerIndex] != playerIndex)
        parent[playerIndex] = findParent(parent[playerIndex], parent);
    return parent[playerIndex];
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int numPlayers;
    cin >> numPlayers;

    // Read player names and skills
    vector<string> playerNames(numPlayers);
    for (int i = 0; i < numPlayers; i++)
        cin >> playerNames[i];

    vector<int> playerSkills(numPlayers);
    for (int i = 0; i < numPlayers; i++)
        cin >> playerSkills[i];

    // Map player's name to their index
    unordered_map<string, int> nameToIndex;
    for (int i = 0; i < numPlayers; i++)
        nameToIndex[playerNames[i]] = i;

    // Union-find setup for friend groups
    int numFriendPairs;
    cin >> numFriendPairs;
    vector<int> parent(numPlayers);
    iota(parent.begin(), parent.end(), 0);

    for (int i = 0; i < numFriendPairs; i++) {
        string nameA, nameB;
        cin >> nameA >> nameB;
        int idxA = findParent(nameToIndex[nameA], parent);
        int idxB = findParent(nameToIndex[nameB], parent);
        if (idxA != idxB) parent[idxB] = idxA;
    }

    // Gather groups of players based on friends
    unordered_map<int, vector<int>> friendGroups;
    for (int i = 0; i < numPlayers; i++)
        friendGroups[findParent(i, parent)].push_back(i);

    // Prepare list of groups, total skill and size per group
    vector<vector<int>> groupMembers;
    vector<int> groupTotalSkill, groupSize;
    for (auto& grp : friendGroups) {
        vector<int>& members = grp.second;
        int totalSkill = 0;
        for (int id : members)
            totalSkill += playerSkills[id];
        groupMembers.push_back(members);
        groupTotalSkill.push_back(totalSkill);
        groupSize.push_back((int)members.size());
    }

    // Rivals
    int numRivalPairs;
    cin >> numRivalPairs;
    int numGroups = groupMembers.size();
    vector<vector<int>> groupRivals(numGroups);

    // For quick lookup from leader/root index to group index
    unordered_map<int, int> rootToGroupIndex;
    {
        int idx = 0;
        for (auto& entry : friendGroups)
            rootToGroupIndex[entry.first] = idx++;
    }

    for (int i = 0; i < numRivalPairs; i++) {
        string nameA, nameB;
        cin >> nameA >> nameB;
        int groupA = rootToGroupIndex[findParent(nameToIndex[nameA], parent)];
        int groupB = rootToGroupIndex[findParent(nameToIndex[nameB], parent)];
        if (groupA != groupB) {
            groupRivals[groupA].push_back(groupB);
            groupRivals[groupB].push_back(groupA);
        }
    }

    int skillLimit;
    cin >> skillLimit;

    // Now check all possible team selections via bitmasking
    int maxTeamSize = 0;
    int maxMask = 1 << numGroups;
    for (int mask = 0; mask < maxMask; mask++) {
        int currentSkill = 0, currentPlayerCount = 0;
        bool validTeam = true;
        // Check for valid rival constraints and skill limit
        for (int groupIdx = 0; groupIdx < numGroups && validTeam; groupIdx++) {
            if (mask & (1 << groupIdx)) {
                // Rival check: If this group has a rival group also in mask, invalid
                for (int rivalIdx : groupRivals[groupIdx]) {
                    if (mask & (1 << rivalIdx)) {
                        validTeam = false;
                        break;
                    }
                }
                currentSkill += groupTotalSkill[groupIdx];
                if (currentSkill > skillLimit) {
                    validTeam = false;
                    break;
                }
                currentPlayerCount += groupSize[groupIdx];
            }
        }
        if (validTeam)
            maxTeamSize = max(maxTeamSize, currentPlayerCount);
    }

    cout << maxTeamSize << "\n";
    return 0;
}
