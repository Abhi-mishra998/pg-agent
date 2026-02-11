/**
 * Recommendation Review UI JavaScript
 * Handles interactive elements, approvals, and audit logging
 */

// ========================================
// Global State
// ========================================

const state = {
    currentUser: null,
    sessionId: null,
    auditLog: [],
    pendingActions: new Set()
};

// ========================================
// Initialization
// ========================================

document.addEventListener('DOMContentLoaded', () => {
    initializeState();
    initializeEventListeners();
    logPageView();
});

function initializeState() {
    // Get session info from page
    const metaElement = document.querySelector('meta[name="session-id"]');
    if (metaElement) {
        state.sessionId = metaElement.getAttribute('content');
    }
    
    // Check for user session (would be replaced with actual auth)
    const userElement = document.querySelector('.user-info');
    if (userElement) {
        state.currentUser = userElement.dataset.user;
    } else {
        state.currentUser = 'anonymous';
    }
}

function initializeEventListeners() {
    // Copy SQL buttons
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', handleCopySQL);
    });
    
    // Action execution buttons
    document.querySelectorAll('.btn-primary').forEach(btn => {
        if (btn.textContent.includes('Execute')) {
            btn.addEventListener('click', handleExecute);
        }
    });
    
    // Approval buttons
    document.querySelectorAll('.btn-success').forEach(btn => {
        if (btn.textContent.includes('Approve')) {
            btn.addEventListener('click', handleApprove);
        }
    });
    
    // Rejection buttons
    document.querySelectorAll('.btn-danger').forEach(btn => {
        if (btn.textContent.includes('Reject')) {
            btn.addEventListener('click', handleReject);
        }
    });
}

// ========================================
// SQL Copy Functionality
// ========================================

function handleCopySQL(event) {
    const button = event.currentTarget;
    const actionId = button.closest('.action-card').id.replace('action-', '');
    const sql = button.dataset.sql;
    
    // Log the copy action
    logAction('copied_sql', { actionId, timestamp: new Date().toISOString() });
    
    // Copy to clipboard
    navigator.clipboard.writeText(sql).then(() => {
        // Visual feedback
        const originalText = button.innerHTML;
        button.innerHTML = '✅ Copied!';
        button.classList.add('copied');
        
        setTimeout(() => {
            button.innerHTML = originalText;
            button.classList.remove('copied');
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy:', err);
        alert('Failed to copy SQL to clipboard');
    });
}

// ========================================
// Action Execution
// ========================================

function handleExecute(event) {
    const button = event.currentTarget;
    const actionCard = button.closest('.action-card');
    const actionId = actionCard.id.replace('action-', '');
    const actionTitle = actionCard.querySelector('.action-title').textContent;
    
    // Safety confirmation for high-risk actions
    const isHighRisk = actionCard.classList.contains('risk-high') || 
                       actionCard.classList.contains('risk-critical');
    
    if (isHighRisk) {
        const confirmed = confirm(
            `⚠️ DANGEROUS ACTION\n\n` +
            `You are about to execute: ${actionTitle}\n\n` +
            `This action has been flagged as HIGH or CRITICAL risk.\n` +
            `Please confirm you have:\n` +
            `• Reviewed the rollback plan\n` +
            `• Notified relevant teams\n` +
            `• Verified this is the correct action\n\n` +
            `Do you want to proceed?`
        );
        
        if (!confirmed) {
            logAction('execution_cancelled', { actionId, reason: 'user_cancelled' });
            return;
        }
    }
    
    // Double confirmation for destructive actions
    const warningsSection = actionCard.querySelector('.warnings-section');
    if (warningsSection && warningsSection.classList.contains('warnings-danger')) {
        const doubleConfirm = confirm(
            `🔴 CRITICAL WARNING\n\n` +
            `This action may cause DATA LOSS or SCHEMA CHANGES.\n\n` +
            `Are you absolutely sure you want to proceed?`
        );
        
        if (!doubleConfirm) {
            logAction('execution_cancelled', { actionId, reason: 'destructive_cancelled' });
            return;
        }
    }
    
    // Execute the action
    executeAction(actionId, button);
}

function executeAction(actionId, button) {
    // Disable button during execution
    button.disabled = true;
    button.innerHTML = '⏳ Executing...';
    
    // Log execution
    logAction('execution_started', { 
        actionId, 
        executor: state.currentUser,
        timestamp: new Date().toISOString() 
    });
    
    // Simulate API call (replace with actual implementation)
    setTimeout(() => {
        // Update button state
        button.innerHTML = '✅ Executed';
        button.classList.remove('btn-primary');
        button.classList.add('btn-success');
        button.disabled = true;
        
        // Update action card status
        const actionCard = document.getElementById(`action-${actionId}`);
        const statusBadge = actionCard.querySelector('.action-status');
        if (statusBadge) {
            statusBadge.textContent = 'EXECUTED';
            statusBadge.classList.add('status-executed');
        }
        
        // Log completion
        logAction('execution_completed', { 
            actionId, 
            executor: state.currentUser,
            timestamp: new Date().toISOString() 
        });
        
        // Show success message
        showNotification('Action executed successfully', 'success');
        
    }, 2000);
}

// ========================================
// Approval Workflow
// ========================================

function handleApprove(event) {
    const button = event.currentTarget;
    const actionCard = button.closest('.action-card');
    const actionId = actionCard.id.replace('action-', '');
    
    // Get approval comment if available
    const comment = prompt('Add an approval comment (optional):');
    
    // Log approval
    logAction('approved', {
        actionId,
        approver: state.currentUser,
        comment: comment || '',
        timestamp: new Date().toISOString()
    });
    
    // Update UI
    updateApprovalUI(actionId, 'approved', state.currentUser);
    
    // Update button state
    const approvalSection = button.closest('.approval-section');
    const statusBadge = approvalSection.querySelector('.approval-status');
    statusBadge.textContent = 'Approved';
    statusBadge.classList.remove('status-pending');
    statusBadge.classList.add('status-approved');
    
    // Replace buttons with success state
    const actionsDiv = approvalSection.querySelector('.approval-actions');
    actionsDiv.innerHTML = '<span class="approval-complete">✅ Approved by ' + state.currentUser + '</span>';
    
    showNotification('Action approved successfully', 'success');
}

function handleReject(event) {
    const button = event.currentTarget;
    const actionCard = button.closest('.action-card');
    const actionId = actionCard.id.replace('action-', '');
    
    // Get rejection reason
    const reason = prompt('Please provide a reason for rejection:');
    if (reason === null) {
        logAction('rejection_cancelled', { actionId, canceller: state.currentUser });
        return;
    }
    
    // Log rejection
    logAction('rejected', {
        actionId,
        rejector: state.currentUser,
        reason: reason,
        timestamp: new Date().toISOString()
    });
    
    // Update UI
    updateApprovalUI(actionId, 'rejected', state.currentUser, reason);
    
    // Update button state
    const approvalSection = button.closest('.approval-section');
    const statusBadge = approvalSection.querySelector('.approval-status');
    statusBadge.textContent = 'Rejected';
    statusBadge.classList.remove('status-pending');
    statusBadge.classList.add('status-rejected');
    
    // Replace buttons with rejection state
    const actionsDiv = approvalSection.querySelector('.approval-actions');
    actionsDiv.innerHTML = '<span class="approval-rejected">❌ Rejected: ' + reason + '</span>';
    
    showNotification('Action rejected', 'warning');
}

function updateApprovalUI(actionId, status, userName, note = '') {
    const actionCard = document.getElementById(`action-${actionId}`);
    const steps = actionCard.querySelectorAll('.approval-step');
    
    // Find the current pending step and update it
    for (let i = 0; i < steps.length; i++) {
        const stepStatus = steps[i].classList.contains('step-approved') || 
                          steps[i].classList.contains('step-rejected');
        
        if (!stepStatus && status === 'approved') {
            steps[i].classList.remove('step-pending');
            steps[i].classList.add('step-approved');
            steps[i].querySelector('.step-marker').textContent = '✅';
            steps[i].querySelector('.step-content').innerHTML += 
                `<div class="step-approver">by ${userName}</div>`;
            steps[i].querySelector('.step-status').textContent = 'Approved';
            break;
        } else if (!stepStatus && status === 'rejected') {
            steps[i].classList.remove('step-pending');
            steps[i].classList.add('step-rejected');
            steps[i].querySelector('.step-marker').textContent = '❌';
            steps[i].querySelector('.step-content').innerHTML += 
                `<div class="step-approver">by ${userName}</div>`;
            steps[i].querySelector('.step-status').textContent = 'Rejected' + (note ? ': ' + note : '');
            break;
        }
    }
}

// ========================================
// Approval Request
// ========================================

function requestApproval(actionId) {
    const actionCard = document.getElementById(`action-${actionId}`);
    const title = actionCard.querySelector('.action-title').textContent;
    
    const approver = prompt(`Request approval for "${title}"\n\nEnter approver role or email:`);
    
    if (approver) {
        logAction('approval_requested', {
            actionId,
            requester: state.currentUser,
            requested_approver: approver,
            timestamp: new Date().toISOString()
        });
        
        showNotification(`Approval request sent to ${approver}`, 'success');
    }
}

// ========================================
// Comments
// ========================================

function addComment(actionId) {
    const actionCard = document.getElementById(`action-${actionId}`);
    const title = actionCard.querySelector('.action-title').textContent;
    
    const comment = prompt(`Add comment for "${title}":`);
    
    if (comment) {
        logAction('comment_added', {
            actionId,
            commenter: state.currentUser,
            comment: comment,
            timestamp: new Date().toISOString()
        });
        
        // Add comment to audit trail
        const auditSection = document.querySelector('.audit-section');
        const timeline = auditSection.querySelector('.audit-timeline');
        
        const newEntry = document.createElement('div');
        newEntry.className = 'audit-entry';
        newEntry.innerHTML = `
            <div class="audit-marker"></div>
            <div class="audit-content">
                <div class="audit-header">
                    <span class="audit-action">Comment</span>
                    <span class="audit-actor">${state.currentUser}</span>
                    <span class="audit-time">${new Date().toISOString()}</span>
                </div>
                <div class="audit-details">${comment}</div>
        `;
        
        timeline.insertBefore(newEntry, timeline.firstChild);
        
        showNotification('Comment added', 'success');
    }
}

// ========================================
// Audit Logging
// ========================================

function logPageView() {
    logAction('page_viewed', {
        user: state.currentUser,
        timestamp: new Date().toISOString(),
        page: window.location.pathname
    });
}

function logAction(actionType, details) {
    const entry = {
        id: generateId(),
        type: actionType,
        ...details,
        sessionId: state.sessionId
    };
    
    state.auditLog.push(entry);
    
    // Also log to console in development
    console.log('[Audit]', actionType, entry);
    
    // In production, would send to audit server
    // sendToAuditServer(entry);
}

// ========================================
// Utility Functions
// ========================================

function generateId() {
    return 'audit_' + Math.random().toString(36).substr(2, 9);
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    
    // Style the notification
    Object.assign(notification.style, {
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        padding: '12px 24px',
        borderRadius: '8px',
        color: '#fff',
        fontWeight: '600',
        zIndex: '10000',
        animation: 'slideIn 0.3s ease'
    });
    
    if (type === 'success') {
        notification.style.background = '#00CC66';
    } else if (type === 'warning') {
        notification.style.background = '#FFAA00';
    } else if (type === 'error') {
        notification.style.background = '#FF4444';
    } else {
        notification.style.background = '#4488FF';
    }
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add animation keyframes
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeOut {
        from {
            opacity: 1;
        }
        to {
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// ========================================
// Export Functions (for testing/integration)
// ========================================

function getAuditLog() {
    return state.auditLog;
}

function exportAuditLog() {
    return JSON.stringify(state.auditLog, null, 2);
}

function getPendingActions() {
    return Array.from(state.pendingActions);
}

// Make functions available globally
window.reviewUI = {
    copySQL: handleCopySQL,
    executeAction: executeAction,
    requestApproval: requestApproval,
    addComment: addComment,
    approveAction: handleApprove,
    rejectAction: handleReject,
    getAuditLog,
    exportAuditLog
};
