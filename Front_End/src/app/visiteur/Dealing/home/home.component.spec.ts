import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { HomevisiteurComponent } from './home.component';

describe('HomeComponent', () => {
  let component: HomevisiteurComponent;
  let fixture: ComponentFixture<HomevisiteurComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ HomevisiteurComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(HomevisiteurComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
