import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { DashboardComponentAdmin } from './dashboard.component';

describe('DashboardComponent', () => {
  let component: DashboardComponentAdmin;
  let fixture: ComponentFixture<DashboardComponentAdmin>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ DashboardComponentAdmin ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(DashboardComponentAdmin);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
