// Minimal behavior to simulate approval flow and audit trail
const timelineEl = document.getElementById('timeline');
const currentUserEl = document.getElementById('current-user');
const btnApprove = document.getElementById('act-approve');
const btnReject = document.getElementById('act-reject');
const btnRequest = document.getElementById('btn-request');
const btnExec = document.getElementById('btn-exec');
const btnCopy = document.getElementById('btn-copy');

let approvals = [];
let requiredApprovers = 0; // For this demo, MEDIUM -> 1 approver

function addTimelineEntry(text){
  const li = document.createElement('li');
  li.textContent = `${new Date().toISOString()} — ${text}`;
  timelineEl.prepend(li);
}

// bootstrap sample
addTimelineEntry('Created by alice (draft)');

btnRequest.addEventListener('click', ()=>{
  addTimelineEntry('Review requested by ' + getCurrentUserName());
  alert('Review requested — reviewers will be notified (simulated)');
});

btnApprove.addEventListener('click', ()=>{
  const who = getCurrentUserName();
  approvals.push({user:who,ts:new Date().toISOString()});
  addTimelineEntry(`Approved by ${who}`);
  checkApprovals();
});

btnReject.addEventListener('click', ()=>{
  const who = getCurrentUserName();
  addTimelineEntry(`Requested changes by ${who}`);
  alert('Reviewer requested changes — author should update the recommendation');
});

btnCopy.addEventListener('click', ()=>{
  const card = document.querySelector('.rec-card');
  const payload = buildSnapshot(card);
  copyToClipboard(JSON.stringify(payload, null, 2));
  addTimelineEntry(`Snapshot copied by ${getCurrentUserName()}`);
  alert('Snapshot prepared for Slack (copied to clipboard if available)');
});

function getCurrentUserName(){
  return currentUserEl.value.split(':')[0];
}

function checkApprovals(){
  // For demo: require 1 approval
  if(approvals.length >= 1){
    btnExec.disabled = false;
    addTimelineEntry('Approval threshold reached — Execute enabled');
  }
}

function buildSnapshot(card){
  const id = card.getAttribute('data-action-id');
  const title = card.querySelector('.title').textContent;
  const sql = card.querySelector('.sql pre code').textContent;
  return { id, title, sql, approvals, timestamp: new Date().toISOString() };
}

function copyToClipboard(text){
  if(navigator.clipboard){
    navigator.clipboard.writeText(text).catch(()=>{
      fallbackDownload(text);
    });
  }else{
    fallbackDownload(text);
  }
}

function fallbackDownload(text){
  const blob = new Blob([text], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'oncall_snapshot.json'; a.click();
  URL.revokeObjectURL(url);
}
